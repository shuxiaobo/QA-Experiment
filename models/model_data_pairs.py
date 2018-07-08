# make sure the model supports the dataset you use
models_in_datasets = {
    "CBT_NE": ["AttentionSumReader", "AoAReader", "Simple_model", "Simple_modelrl", "Simple_model1"],
    "CBT_CN": ["AttentionSumReader", "AoAReader", "Simple_model", "Simple_modelrl", "Simple_model1"],
    "SQuAD": ["RNet", "SimpleModelSQuad", "SimpleModelSQuad2", "SimpleModelSQuad3", "SimpleModelSQuad4", "SimpleModelSQuadBiDAF", "SimpleModelSQuad5"]
}
