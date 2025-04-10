/*
 * @vulnerable_at_lines: 17
 */


pragma solidity ^0.4.24;

contract ShortAddressVulnerable4 {
    mapping(address => uint256) public balances;

    event CoinSent(address indexed sender, address indexed receiver, uint256 amount);

    function depositCoins() public payable {
        balances[msg.sender] += msg.value;
    }

    function sendCoin(address receiver, uint amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[receiver] += amount;
        emit CoinSent(msg.sender, receiver, amount);
    }

    function getUserBalance(address user) public view returns (uint256) {
        return balances[user];
    }
}