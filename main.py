import sys

from models.nlp_base import NLPBase

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,2" # Note: if set "0,1,2,3" and the #1 GPU is using, will cause OOM Error

def get_model_class(model_name):
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            return NLPBase()
        class_obj, class_name = None, sys.argv[1]
    else:
        class_obj, class_name = None, None
    try:
        import models
        class_obj = getattr(sys.modules["models"], class_name if model_name == None else model_name)
        # sys.argv.pop(1)
    except AttributeError or IndexError:
        print("Model [{}] not found.\nSupported models:\n\n\t\t{}\n".format(class_name, sys.modules["models"].__all__))
        exit(1)
    return class_obj()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
    model = get_model_class('SimpleModelSQuad3')
    model.execute()
