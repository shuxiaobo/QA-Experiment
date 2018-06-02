# make sure the model supports the dataset you use
models_in_datasets = {
    "CBT_NE": ["AttentionSumReader", "AoAReader", "Simple_model", "Simple_modelrl"],
    "CBT_CN": ["AttentionSumReader", "AoAReader", "Simple_model", "Simple_modelrl"],
    "SQuAD": ["RNet"]
}
