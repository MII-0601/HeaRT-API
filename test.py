def create_item(item):
	# item.text
	dir_file = open('test.txt', 'w', encoding='UTF-8')
	dir_file.writelines(item)
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
    pred_dir = "tmp/"
    batch_size = 1
    test_file = "/home/is/mihiro-n/JaMIE-main/output_conll2/single.conll"
    test_output = "/home/is/mihiro-n/JaMIE-main/output"

    test_comments, test_toks, test_ners, test_mods, test_rels, _, _, _, _ = utils.extract_rel_data_from_mh_conll_v2(
                test_file,
                down_neg=0.0)

            max_len = utils.max_sents_len(test_toks, tokenizer)
            cls_max_len = max_len + 2

            test_dataset, test_comment, test_tok, test_ner, test_mod, test_rel, test_spo = utils.convert_rels_to_mhs_v3(
                test_comments, test_toks, test_ners, test_mods, test_rels,
                tokenizer, bio2ix, mod2ix, rel2ix, cls_max_len, verbose=0)

            cls_max_len = min(cls_max_len, bert_max_len)

            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            eval_joint(model, test_dataloader, test_comment, test_tok, test_ner, test_mod, test_rel, test_spo,
                       bio2ix, mod2ix, rel2ix, cls_max_len, args.device, "Final test dataset",
                       print_levels=(2, 2, 2), out_file=args.test_output, test_mode=False, verbose=0)
            test_evaluator = MhsEvaluator(args.test_file, args.test_output)
            test_evaluator.eval_ner(print_level=1)
            test_evaluator.eval_mod(print_level=1)
            # test_evaluator.eval_rel(print_level=2)
            test_evaluator.eval_mention_rel(print_level=2)

    # conll2xml
    # conll_dir,xml_dir
    output_conll = "output_cl"
    xml_dir = "./XML"
    conll_to_xml(output_conll,xml_dir)

    return xml_dir

item = "テスト"
create_item(item)
