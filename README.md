# End-to-End-Transition-Based-Online-Dialogue-Disentanglement

Source code and dataset for the IJCAI 2020 paper "End-to-End Transition-Based Online Dialogue Disentanglement", which aims to solve the problem of dialogue disentanglement in an end-to-end manner. This work achieves better results comparing with all the previous methods, which mostly adopt a two-step architecture. You can download the paper from [here](https://www.ijcai.org/proceedings/2020/0535.pdf).

## Dataset

Our proposed Movie Dialogue Dataset is contained in the ``data/`` folder. 
The format of each sample is as:

	[["speaker": "xxx", "utterance": "xxx", "dialogue_id": "xx", "section_id": "xx", "label": "x"], 
	 ["speaker": "xxx", "utterance": "xxx", "dialogue_id": "xx", "section_id": "xx", "label": "x"] ... ]

For each utterance, "speaker" indicates the speaker of the utterance (which is not used in this work), and "utterance" is the content of the current utterance. "label" indicates which session the current utterance belongs to.

## Code

The code for our proposed end-to-end framework is contained in the ``code/`` folder.

Before you run the code, you have to specify some hyperparameters and some file paths in the file ``constant.py`` .

You can train the model by the following command:

	python main.py --mode train --input_path [path of the data] --device [cpu or gpu index]

After train the model, the test command is as:

	python main.py --mode test --input_path [path of the data] --model_path [path to the model you want to test] --device [cpu or gpu index]

## Reference

Please cite the paper in the following format if you use this dataset during your research.

	@inproceedings{ijcai2020-535,
      title     = {End-to-End Transition-Based Online Dialogue Disentanglement},
      author    = {Liu, Hui and Shi, Zhan and Gu, Jia-Chen and Liu, Quan and Wei, Si and Zhu, Xiaodan},
      booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
                   Artificial Intelligence, {IJCAI-20}},
      publisher = {International Joint Conferences on Artificial Intelligence Organization},             
      editor    = {Christian Bessiere}  
      pages     = {3868--3874},
      year      = {2020},
      month     = {7},
      note      = {Main track}
      doi       = {10.24963/ijcai.2020/535},
      url       = {https://doi.org/10.24963/ijcai.2020/535},
    }


## Miscellaneous

If you find any problem about the code, please leave an issue or shoot me an email.

