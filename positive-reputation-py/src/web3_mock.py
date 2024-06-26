EOA_ADDRESS_NEW = '0xA1B4355Ae6b39bb403Be1003b7D0330C811747DB'
EOA_ADDRESS_OLD = '0x1c5dCdd006EA78a7E4783f9e6021C30000000000'
EOA_ADDRESS_CONTRACT_DEPLOYER = '0x9C33eaCc2F50E39940D3AfaF2c7B8246B681A374'


class Web3Mock:
    def __init__(self):
        self.eth = EthMock()


class EthMock:
    chain_id = 1

    def __init__(self):
        self.contract = ContractMock()

    def get_transaction_count(self, address: str, block_identifier):
        if address.lower() == EOA_ADDRESS_NEW.lower():
            return 0
        elif address.lower() == EOA_ADDRESS_OLD.lower():
            return 501
        else:
            raise ValueError('Unknown address')


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
