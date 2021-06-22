from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from utils import * 

app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"status":Failure,"detail": exc.errors(), "body": exc.body})
    )


class Item(BaseModel):
    status: Optional[str] = "Success"
    text: str


@app.post("/kogecha.naist.jp/prism-model/parse")
async def create_item(item: Item):
	#item.text
	dir_file = open('test.txt','w',encoding='UTF-8')
	dir_file.writelines(item.text)
	dir_file.close()
	
	file_name = dir_file.split('/')[-1].rsplit('.', 1)[0]
	conll_dir = './output_conll'

	single_convert_document_to_conll(
		dir_file,
		os.path.join(
			conll_dir,
			f"{file_name}.conll"
			),
		sent_tag=True,
                contains_modality=True,
                with_dct=with_dct,
                is_raw=is_raw,
                morph_analyzer_name=segmenter,
                bert_tokenizer=bert_tokenizer,
                is_document=True
		)   
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	n_gpu = torch.cuda.device_count()

	saved_model = "./ouall"
	
	tokenizer = BertTokenizer.from_pretrained(
            saved_model,
            do_lower_case=args.do_lower_case,
            do_basic_tokenize=False,
            tokenize_chinese_chars=False
        )
        with open(os.path.join(saved_model, 'ner2ix.json')) as json_fi:
            bio2ix = json.load(json_fi)
        with open(os.path.join(saved_model, 'mod2ix.json')) as json_fi:
            mod2ix = json.load(json_fi)
        with open(os.path.join(saved_model, 'rel2ix.json')) as json_fi:
            rel2ix = ccon.load(json_fi)
	model = torch.load(os.path.join(saved_model, 'model.pt'))
        model.to(device)

	test_dir = "tmp/"

