import torch
import warnings
import forta_agent
import pandas as pd
from forta_agent import get_json_rpc_url, EntityType
from joblib import load
from evmdasm import EvmBytecode
from web3 import Web3
from os import environ

from transformers import (
                    AutoConfig,
                    AutoTokenizer,
                    AutoModelForCausalLM,
                    PretrainedConfig,
                    )

from src.constants import (
    BYTE_CODE_LENGTH_THRESHOLD,
    MODEL_THRESHOLD,
    SAFE_CONTRACT_THRESHOLD,
)
from src.findings import ContractFindings
from src.logger import logger
from src.utils import (
    calc_contract_address,
    get_features,
    get_function_signatures,
    get_storage_addresses,
    is_contract,
)
from src.storage import get_secrets

SECRETS_JSON = get_secrets()

web3 = Web3(Web3.HTTPProvider(get_json_rpc_url()))
ML_MODEL = None
TOKENIZER = None

# Supress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = './model'
tokenizer_path = './tokenizer'

# Anomaly parameters
mean = 1.9269466400146484
std_dev = 1.5235518217086792
delta = 3
anomaly_threshold = mean + delta * std_dev
stride = 128

labels_ids = {'malicious': 0, 'normal': 1}
n_labels = len(labels_ids)

def initialize():
    """
    this function loads the ml model.
    """
    global ML_MODEL
    global TOKENIZER
    logger.info("Loading configuration...")
    # Get model configuration.
    model_config = AutoConfig.from_pretrained(model_path)
    # Get model's tokenizer.
    logger.info('Loading tokenizer...')
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path)
    # default to left padding
    TOKENIZER.padding_side = "left"
    # Define PAD Token = EOS Token = 0
    TOKENIZER.pad_token = TOKENIZER.eos_token
    # Get the actual model.
    logger.info('Loading model...')
    ML_MODEL = AutoModelForCausalLM.from_pretrained(model_path, config=model_config)
    # resize model embedding to match new tokenizer
    ML_MODEL.resize_token_embeddings(len(TOKENIZER))
    # fix model padding token id
    ML_MODEL.config.pad_token_id = ML_MODEL.config.eos_token_id
    # Load model to defined device.
    ML_MODEL.to(device)
    logger.info("Complete loading model")

    global CHAIN_ID
    CHAIN_ID = web3.eth.chain_id

    environ["ZETTABLOCK_API_KEY"] = SECRETS_JSON["apiKeys"]["ZETTABLOCK"]


def exec_model(w3, opcodes: str, contract_creator: str) -> tuple:
    """
    this function executes the model to obtain the score for the contract
    :return: score: float
    """
    score = None
    features, opcode_addresses = get_features(w3, opcodes, contract_creator)
    score = evaluate_sample(features, contract_creator, stride, device)

    return score, opcode_addresses

def evaluate_sample(opcodes, contract_creator, stride, device):
    global anomaly_threshold
    global ML_MODEL
    global labels_ids
    ML_MODEL.eval()
    endoftext = 0

    processed_batch = {}
    encodings = TOKENIZER(str(opcodes), return_tensors="pt")
    max_length = ML_MODEL.config.n_positions
    seq_len = encodings.input_ids.size(1)
    ML_MODEL.eval()
    nlls = []
    prev_end_loc = 0
    anomalous_sequences = []
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        list = input_ids[0].tolist()
        # padding to the left
        list[0:0] = [endoftext] * (max_length - len(list))
        input_ids = torch.tensor(list, dtype=torch.long).view(1, max_length).to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = ML_MODEL(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            local_perplexity = torch.exp(neg_log_likelihood)
            if local_perplexity > anomaly_threshold:
                return labels_ids['malicious']
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return labels_ids['normal']

def detect_malicious_contract_tx(
    w3, transaction_event: forta_agent.transaction_event.TransactionEvent
) -> list:
    malicious_findings = []
    safe_findings = []


    for trace in transaction_event.traces:
        if trace.type == "create":
            created_contract_address = (
                trace.result.address if trace.result else None
            )
            error = trace.error if trace.error else None
            logger.info(f"Contract created {created_contract_address}")
            if error is not None:
                if transaction_event.from_ == trace.action.from_:
                    nonce = transaction_event.transaction.nonce
                    contract_address = calc_contract_address(w3, trace.action.from_, nonce)
                else:
                    # For contracts creating other contracts, get the nonce using Web3
                    nonce = w3.eth.getTransactionCount(Web3.toChecksumAddress(trace.action.from_), transaction_event.block_number)
                    contract_address = calc_contract_address(w3, trace.action.from_, nonce - 1)

                logger.warn(
                    f"Contract {contract_address} creation failed with tx {trace.transactionHash}: {error}"
                )

            # creation bytecode contains both initialization and run-time bytecode.
            creation_bytecode = trace.action.init
            for finding in detect_malicious_contract(
                w3,
                trace.action.from_,
                created_contract_address,
                creation_bytecode,
                error=error,
            ):
                if finding.alert_id == "SUSPICIOUS-CONTRACT-CREATION":
                    malicious_findings.append(finding)
                else:
                    safe_findings.append(finding)

    if transaction_event.to is None:
        nonce = transaction_event.transaction.nonce
        created_contract_address = calc_contract_address(
            w3, transaction_event.from_, nonce
        )
        logger.info(f"Contract created {created_contract_address}")
        creation_bytecode = transaction_event.transaction.data
        for finding in detect_malicious_contract(
            w3,
            transaction_event.from_,
            created_contract_address,
            creation_bytecode,
        ):
            if finding.alert_id == "SUSPICIOUS-CONTRACT-CREATION":
                malicious_findings.append(finding)
            else:
                safe_findings.append(finding)

    # Reduce findings to 10 because we cannot return more than 10 findings per request
    return (malicious_findings + safe_findings)[:10]


def detect_malicious_contract(
    w3, from_, created_contract_address, code, error=None
) -> list:
    findings = []

    if created_contract_address is not None and code is not None:
        if len(code) > BYTE_CODE_LENGTH_THRESHOLD:
            try:
                opcodes = EvmBytecode(code).disassemble()
            except Exception as e:
                logger.warn(f"Error disassembling evm bytecode: {e}")
            # obtain all the addresses contained in the created contract and propagate to the findings
            storage_addresses = get_storage_addresses(w3, created_contract_address)
            (
                model_score,
                opcode_addresses,
            ) = exec_model(w3, opcodes, from_)
            function_signatures = get_function_signatures(w3, opcodes)
            logger.info(f"{created_contract_address}: score={model_score}")

            finding = ContractFindings(
                from_,
                created_contract_address,
                set.union(storage_addresses, opcode_addresses),
                function_signatures,
                model_score,
                MODEL_THRESHOLD,
                error=error,
            )
            if model_score is not None:
                from_label_type = "contract" if is_contract(w3, from_) else "eoa"
                labels = [
                    {
                        "entity": created_contract_address,
                        "entity_type": EntityType.Address,
                        "label": "contract",
                        "confidence": 1.0,
                    },
                    {
                        "entity": from_,
                        "entity_type": EntityType.Address,
                        "label": from_label_type,
                        "confidence": 1.0,
                    },
                ]

                if model_score == labels_ids["malicious"]:
                    labels.extend(
                        [
                            {
                                "entity": created_contract_address,
                                "entity_type": EntityType.Address,
                                "label": "attacker",
                                "confidence": model_score,
                            },
                            {
                                "entity": from_,
                                "entity_type": EntityType.Address,
                                "label": "attacker",
                                "confidence": model_score,
                            },
                        ]
                    )

                    findings.append(
                        finding.malicious_contract_creation(
                            CHAIN_ID,
                            labels,
                        )
                    )
                elif model_score == labels_ids["normal"]:
                    labels.extend(
                        [
                            {
                                "entity": created_contract_address,
                                "entity_type": EntityType.Address,
                                "label": "positive_reputation",
                                "confidence": 1 - model_score,
                            },
                            {
                                "entity": from_,
                                "entity_type": EntityType.Address,
                                "label": "positive_reputation",
                                "confidence": 1 - model_score,
                            },
                        ]
                    )
                    findings.append(
                        finding.safe_contract_creation(
                            CHAIN_ID,
                            labels,
                        )
                    )
                else:
                    findings.append(finding.non_malicious_contract_creation(CHAIN_ID))

    return findings


def provide_handle_transaction(w3):
    def handle_transaction(
        transaction_event: forta_agent.transaction_event.TransactionEvent,
    ) -> list:
        return detect_malicious_contract_tx(w3, transaction_event)

    return handle_transaction


real_handle_transaction = provide_handle_transaction(web3)


def handle_transaction(
    transaction_event: forta_agent.transaction_event.TransactionEvent,
):
    return real_handle_transaction(transaction_event)
