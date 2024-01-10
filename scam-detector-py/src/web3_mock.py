from hexbytes import HexBytes

EOA_ADDRESS_SMALL_TX = '0x1c5dCdd006EA78a7E4783f9e6021C32935a10fb4'  # small tx count
EOA_ADDRESS_LARGE_TX = '0xdec08cb92a506B88411da9Ba290f3694BE223c26'  # large tx count
FUTURE_CONTRACT_TO_BE_DEPLOYED_BY_EOA_LARGE_TX = '0x85ce07420955881ec4d12538d77e3370cd163539'
CONTRACT = '0x2320A28f52334d62622cc2EaFa15DE55F9987eD9'
CONTRACT2 = '0x440aECA896009f006EEA3df4BA3A236EE8D57D36'
SCAM_CONTRACT_DEPLOYER = '0xADE2E990D714D8118814ea998a4B9b4160a74741'



class Web3Mock:
    def __init__(self):
        self.eth = EthMock()


class EthMock:
    def __init__(self):
        self.contract = ContractMock()

    def get_transaction_count(self, address, block_identifier=None):
        if block_identifier is None:
            if address == EOA_ADDRESS_SMALL_TX:
                return 1999
            elif address == EOA_ADDRESS_LARGE_TX:
                return 2001
        else:
            if address == SCAM_CONTRACT_DEPLOYER:
                return 53
        return 0

    def chain_id(self):
        return 1

    def get_transaction(self, hash):
        if hash == '0x53244cc27feed6c1d7f44381119cf14054ef2aa6ea7fbec5af4e4258a5a026aa':
            return {'to': '0x91C1B58F24F5901276b1F2CfD197a5B73e31F96E'}
        else:
            return {'to': '0x21e13f16838e2fe78056f5fd50251ffd6e7098b4'}

    def get_code(self, address):
        if address == EOA_ADDRESS_SMALL_TX:
            return HexBytes('0x')
        elif address == CONTRACT or address == CONTRACT2:
            return HexBytes('0x0000000000000000000000000000000000000000000000000000000000000005')
        elif address.lower() == '0xd89a4f1146ea3ddc8f21f97b562d97858d89d307':
            return HexBytes('0x608060405261138860005534801561001657600080fd5b5060006040516024016040516020818303038152906040527f8129fc1c000000000000000000000000000000000000000000000000000000007bffffffffffffffffffffffffffffffffffffffffffffffffffffffff19166020820180517bffffffffffffffffffffffffffffffffffffffffffffffffffffffff83818316178352505050509050600073365865997b9851f7e32afc16299facaa5b25824873ffffffffffffffffffffffffffffffffffffffff16826040516100d991906101de565b600060405180830381855af49150503d8060008114610114576040519150601f19603f3d011682016040523d82523d6000602084013e610119565b606091505b505090508061015d576040517f08c379a000000000000000000000000000000000000000000000000000000000815260040161015490610252565b60405180910390fd5b5050610272565b600081519050919050565b600081905092915050565b60005b8381101561019857808201518184015260208101905061017d565b838111156101a7576000848401525b50505050565b60006101b882610164565b6101c2818561016f565b93506101d281856020860161017a565b80840191505092915050565b60006101ea82846101ad565b915081905092915050565b600082825260208201905092915050565b7f56657269666963696174696f6e2e000000000000000000000000000000000000600082015250565b600061023c600e836101f5565b915061024782610206565b602082019050919050565b6000602082019050818103600083015261026b8161022f565b9050919050565b6101ee806102816000396000f3fe60806040523661000b57005b600036606060008073365865997b9851f7e32afc16299facaa5b25824873ffffffffffffffffffffffffffffffffffffffff16858560405161004e929190610122565b600060405180830381855af49150503d8060008114610089576040519150601f19603f3d011682016040523d82523d6000602084013e61008e565b606091505b5091509150816100d3576040517f08c379a00000000000000000000000000000000000000000000000000000000081526004016100ca90610198565b60405180910390fd5b8092505050915050805190602001f35b600081905092915050565b82818337600083830152505050565b600061010983856100e3565b93506101168385846100ee565b82840190509392505050565b600061012f8284866100fd565b91508190509392505050565b600082825260208201905092915050565b7f566572696669636174696f6e2e00000000000000000000000000000000000000600082015250565b6000610182600d8361013b565b915061018d8261014c565b602082019050919050565b600060208201905081810360008301526101b181610175565b905091905056fea264697066735822122098fcbe5df46358c09902e35b24514e53c1638b5bfbd66bb9cd07f85b970bae0264736f6c634300080d0033')
        else:
            return HexBytes('0x')


class ContractMock:
    def __init__(self):
        self.functions = FunctionsMock()

    def __call__(self, address, *args, **kwargs):
        return self


class FunctionsMock:
    def __init__(self):
        self.return_value = None

    def call(self, *_, **__):
        return self.return_value
