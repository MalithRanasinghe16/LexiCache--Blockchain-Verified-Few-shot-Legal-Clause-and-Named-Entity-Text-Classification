from web3 import Web3

rpc_url = "https://sepolia.infura.io/v3/4ae413eca67e451395a2b947e95f5ba9" 
w3 = Web3(Web3.HTTPProvider(rpc_url))
print("Connected to Sepolia:", w3.is_connected())