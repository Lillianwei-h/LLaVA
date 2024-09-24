from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import json
from tqdm import tqdm
from mydata import get_data, decode_answer
import os

DATA_PATH  = "../../data/ManytoMany_all"
model_path = "liuhaotian/llava-v1.6-34b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

for dataset in ['mathvista','vist', 'wikihow']:
    os.makedirs(dataset,exist_ok=True)
    data_path = os.path.join(DATA_PATH,dataset,'data.json')
    temp_output_path = f'./{dataset}/llava_results_temp.json'
    output_path = f'./{dataset}/llava_results.json'
    print(dataset)

    content_dict = get_data(data_path, dataset)
    for key, value in tqdm(content_dict.items()):
        args = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "query": value['prompt'],
            "conv_mode": None,
            "image_file": value['image_file'],
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()
        output = eval_model(args,tokenizer, model, image_processor, context_len)
        value['gpt'] = output
        value.pop("prompt")
        value.pop("image_file")

    with open(output_path, 'w') as f:
        answer_list = [value for _,value in content_dict.items()]
        json.dump(answer_list, f, indent = 4)

    print(f"Save output to: {output_path}")