# open final_dataset_with_line_numbers_v2.csv and import to pandas dataframe
from typing import List
import pandas as pd

df = pd.read_csv("./datasets/processed/solidity/v2/final_dataset_with_line_numbers_v2.csv")

# print the first 5 rows of the dataframe
print(df.head())

# print the shape of the dataframe
print(df.shape)

# print the column names
print(df.columns)

CLASSIFICATION_PROMPT = f"Analyze the following Solidity smart contract and classify it into one or more of the following vulnerability types based on its most critical security flaw: access_control, bad_randomness, delegatecall, denial_of_service, front_running, integer_overflow_underflow, non-vulnerable, numerical_consistency, reentrancy, short_addresses, timestamp_dependency, transaction_ordering_dependency, unchecked_call, unprotected self-destruct and other. Return only the appropriate vulnerability types from the list above."

LOCALIZATION_PROMPT = f"Smart contracts written in Solidity language may contain vulnerabilities such as DelegateCall, Arithmetic/Integer Overflow and Underflow, Nested Call, Reentrancy, Timestamp Dependency, TxOrigin, Transaction Order Dependency, Unchecked Call, Unprotected Suicide, Frozen Ether, Bad Randomness, Denial of service, Front Running, Short Address and other vulnerabilities. Analyze the following solidity smart contract for security vulnerabilities, bugs, and faulty logic. Identify all problematic lines and explain the risks associated with each."

# send a prompt to claude sonnet
import anthropic
import openai
from dotenv import load_dotenv
load_dotenv()
import instructor

from pydantic import BaseModel
from enum import Enum

# access_control, bad_randomness, delegatecall, denial_of_service, front_running, integer_overflow_underflow, non-vulnerable, numerical_consistency, reentrancy, short_addresses, timestamp_dependency, transaction_ordering_dependency, unchecked_call, 
# unprotected self-destruct and other. Return only the appropriate vulnerability types from the list above."
class VulnerabilityType(Enum):
    ACCESS_CONTROL = "access_control"
    BAD_RANDOMNESS = "bad_randomness"
    DELEGATECALL = "delegatecall"
    DENIAL_OF_SERVICE = "denial_of_service"
    FRONT_RUNNING = "front_running"
    INTEGER_OVERFLOW_UNDERFLOW = "integer_overflow_underflow"
    NON_VULNERABLE = "non-vulnerable"
    NUMERICAL_CONSISTENCY = "numerical_consistency"
    REENTRANCY = "reentrancy"
    SHORT_ADDRESSES = "short_addresses"
    TIMESTAMP_DEPENDENCY = "timestamp_dependency"
    TRANSACTION_ORDERING_DEPENDENCY = "transaction_ordering_dependency"
    UNCHECKED_CALL = "unchecked_call"
    UNPROTECTED_SELF_DESTRUCT = "unprotected_self_destruct"
    OTHER = "other" # if the vulnerability is not in the list above

class VulnerableLines(BaseModel):
    vulnerable_lines: list[int]
    description: str


claude_client = instructor.from_anthropic(anthropic.Anthropic())
openai_client = instructor.from_openai(openai.OpenAI())

generic_columns = ['classification_prompt', 'localization_prompt', 'source_code', 'vulnerability_type', 'vulnerable_lines', 'vulnerable_lines_description']

try:
    # open the temp files as seperate dfs
    claude_df = pd.read_csv('./temp/claude-sonnet-4-20250514_prompt.csv', delimiter=',')
    openai_df = pd.read_csv('./temp/gpt-4.1-2025-04-14_prompt.csv', delimiter=',')
except Exception as e:
    print(e)
    claude_df = pd.DataFrame(columns=generic_columns)
    openai_df = pd.DataFrame(columns=generic_columns)

print("\n"*5)
print(claude_df.head())
print(openai_df.head())
print("\n"*5)

async def send_prompt(client, model, source_code):
    # check if we have this in dfs
    try:
        if model == "claude-sonnet-4-20250514":
            selected_df = claude_df
        elif model == "gpt-4.1-2025-04-14":
            selected_df = openai_df
        else:
            # classification_prompt, localization_prompt, source_code, vulnerability_type, vulnerable_lines, vulnerable_lines_description
            selected_df = pd.DataFrame(columns=generic_columns)
        if source_code in selected_df['source_code']:
            print(f"Found in {model} df")
            # return vulnerability_type, vulnerable_lines
            vulnerable_lines = VulnerableLines(vulnerable_lines=selected_df[selected_df['source_code'] == source_code]['vulnerable_lines'].values[0], description=selected_df[selected_df['source_code'] == source_code]['vulnerable_lines_description'].values[0])
            vulnerability_type = selected_df[selected_df['source_code'] == source_code]['vulnerability_type'].values[0]
            print(vulnerability_type)
            print(vulnerable_lines.vulnerable_lines)
            print(vulnerable_lines.description)
            return vulnerability_type, vulnerable_lines
    except Exception as e:
        print(e)
    
    async def get_vulnerability_type(model):
        return client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": CLASSIFICATION_PROMPT, "cache_control": {"type": "ephemeral"}},
                {"role": "user", "content": source_code}
            ],
            response_model=List[VulnerabilityType]
        )
    
    async def get_vulnerable_lines(model):
        return client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": LOCALIZATION_PROMPT, "cache_control": {"type": "ephemeral"}},
                {"role": "user", "content": source_code}
            ],
            # enable prompt caching
            response_model=VulnerableLines
        )
    
    vulnerability_type, vulnerable_lines = await asyncio.gather(
        get_vulnerability_type(model),
        get_vulnerable_lines(model)
    )
    # save the prompt and response to selected_df
    selected_df = pd.concat([selected_df, pd.DataFrame({
        'classification_prompt': CLASSIFICATION_PROMPT,
        'localization_prompt': LOCALIZATION_PROMPT,
        'source_code': source_code,
        'vulnerability_type': vulnerability_type.value,
        'vulnerable_lines': vulnerable_lines.vulnerable_lines,
        'vulnerable_lines_description': vulnerable_lines.description
    })])
    # add a new row

    # save the df to a csv file
    selected_df.to_csv(f'temp/{model}_prompt.csv', index=False)


    return vulnerability_type, vulnerable_lines

import tqdm
#iterate through the each row in the df and get the source code

async def main():
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        source_code = row['source_code']
        claude_vulnerability_type, claude_vulnerable_lines = await send_prompt(claude_client, "claude-sonnet-4-20250514", source_code)

        # add the vulnerability type and vulnerable lines to the df
        df.at[index, 'claude_vulnerability_type'] = claude_vulnerability_type
        df.at[index, 'claude_vulnerable_lines'] = ",".join([str(x) for x in claude_vulnerable_lines.vulnerable_lines])
        df.at[index, 'claude_vulnerable_lines_description'] = claude_vulnerable_lines.description

        # send a prompt to openai
        
        openai_vulnerability_type, openai_vulnerable_lines = await send_prompt(openai_client, "gpt-4.1-2025-04-14", source_code)

        # add the vulnerability type and vulnerable lines to the df
        df.at[index, 'openai_vulnerability_type'] = openai_vulnerability_type
        df.at[index, 'openai_vulnerable_lines'] = ",".join([str(x) for x in openai_vulnerable_lines.vulnerable_lines])
        df.at[index, 'openai_vulnerable_lines_description'] = openai_vulnerable_lines.description

    # save the df to a csv file
    # create a new directory if it doesn't exist
    import os
    os.makedirs('./datasets/evaluations/solidity/v2', exist_ok=True)
    df.to_csv('./datasets/evaluations/solidity/v2/claude_openai_outputs.csv', index=False)
import asyncio
if __name__ == "__main__":
    asyncio.run(main())