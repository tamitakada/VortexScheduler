# Azure Public Trace

Azure public trace github repo: https://github.com/Azure/AzurePublicDataset

Using Azure function and LLM inference request trace
https://github.com/Azure/AzurePublicDataset/tree/master/data 


The csv files on github is trimmed trace (currently we only upload a subset of them top 5000 rows) due to github file size limit, if want full data, one can download them and run the script:
- function trace: https://github.com/Azure/AzurePublicDataset/blob/master/data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar 

```python function_rar_preprocess_trace.py```

- LLM trace: https://github.com/Azure/AzurePublicDataset/blob/master/data/AzureLLMInferenceTrace_code.csv

```python llm_csv_preprocess_trace.py ```

By setting the TOP_NUM_ROWS_TO_RETRIEVE=None, it will produce the trace for all data points