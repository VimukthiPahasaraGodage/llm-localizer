1. First tokenize the inputs(source codes) for each of the LLM models that will be used
2. Then inference the last hidden state for the tokenized tensors for each of the LLM models that will be used
3. The generate configurations for the experiments to be conducted and run the localization_model train loop

How to start the Ray cluster
```
ray start --head --port=6379 --dashboard-host=0.0.0.0
```

```commandline
huggingface-cli download Salesforce/codegen-350M-mono --repo-type model
```
Then run the python file

defects4j
v1 - cleaned source code
v2 - cleaned source code, target lines which are comments removed
v3 - original cleaned code, target lines which are comments removed