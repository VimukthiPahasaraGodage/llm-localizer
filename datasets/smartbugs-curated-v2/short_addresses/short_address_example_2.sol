/*
 * @vulnerable_at_lines: 16
 */

pragma solidity ^0.4.24;

contract ShortAddressVulnerable2 {
    mapping(address => uint256) public balances;

    event Sent(address indexed from, address indexed to, uint256 amount);

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function sendFunds(address to, uint amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[to] += amount;
        emit Sent(msg.sender, to, amount);
    }

    function checkBalance(address user) public view returns (uint256) {
        return balances[user];
    }
}