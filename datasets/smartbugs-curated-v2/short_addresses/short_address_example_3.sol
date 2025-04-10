/*
 * @vulnerable_at_lines: 17
 */


pragma solidity ^0.4.24;

contract ShortAddressVulnerable3 {
    mapping(address => uint256) public balances;

    event TokenTransferred(address indexed from, address indexed to, uint256 value);

    function depositTokens(uint256 amount) public {
        balances[msg.sender] += amount;
    }

    function transferTokens(address recipient, uint256 value) public {
        require(balances[msg.sender] >= value, "Insufficient balance");
        balances[msg.sender] -= value;
        balances[recipient] += value;
        emit TokenTransferred(msg.sender, recipient, value);
    }

    function balanceOf(address user) public view returns (uint256) {
        return balances[user];
    }
}