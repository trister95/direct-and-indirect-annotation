# **Direct and Indirect Annotation with Generative AI: A Case Study into Finding Animals and Plants in Historical Text**
Here, you will find the repository for the paper "Direct and Indirect Annotation with Generative AI: A Case Study into Finding Animals and Plants in Historical Text" as (to be) presented on the Computational Humanities Research Conference in Aarhus, Denmark, 4-6 December, 2024. When the paper is published in the conference proceedings, a link will be placed here.

In our study, we compared indirect GenAI annotation with direct GenAI annotation and human annotations. For indirect annotation we adopted the LLMaAA framework, introduced by Zhang et al. (2023). We used their code as starting point, stripped it down for the purpose of our research, and also made some changes. Most notably: 

* we work with the OpenAI API instead of Azure;
* we build in an integration with the Huggingface ecoystem;
* we adjusted for package updates. 

It should be noted that there are still a lot of remnants that have more to do with their project than theirs (e.g. Chinese text, name of datasets). We did not make any attempts to systematically remove these, because our goal here was just to study the use of their method in humanities research. We have tried to make the code intelligible, but there will probably be places where the code could have been cleaner, more pythonic, etc. 

If you are interested in using this method, we advise you to also inspect the [LLMaAA paper by Zhang et al. (2023)](https://aclanthology.org/2023.findings-emnlp.872n) and the accompanying repository [the accompanying repository](https://github.com/ridiculouz/LLMaAA/tree/main). Although we have added things ourselves, this repository used theirs as a starting point, as can be easily seen in the structure of the repo as well as the code itself. Note that you will have to insert your own OpenAI API key to work with this code.

Regarding the last bullet point: the GenAI field is very dynamic and it is likely that some further changes are needed due to updated packages. Changes in LangChain and the OpenAI ecosystem make some parts of our code redundant (e.g., parts of the safeguard we build into the annotation scripts are probably not needed when using the new "structured outputs" option). We, therefore, also advice to critically evaluate the existing code before implementation. It is more than likely that new developments make implementation simplier. 

## **Repo structure**
The structure of this repository is as follows:
* Annotations: in this folder annotations from the two human annotars are stored. Those annotations can be used by other researchers to train their own models. We are planning on releasing these and other annotations in a more structured way. However, the human annotation process is still partly ongoing, so the official release of such a dataset should not be expected before the second half of 2025.
* Data: data used for the training of the various models is stored in this folder. Also, the predictions with direct annotations are saved here.
The subfolders have names that start with "by_the_horns", which was for some time the working title of the project. When the name changed to the current one, those folder names were not changed to prevent breaking paths (etc.). We hope it doesn't confuse readers too much. The folder also has a python script that can be used to preprocess data for the NER task (/token classification).
* Logs: this folder is empty. When using LLMaAA logs of the training process show up here.
* Results: subfolders are quiet self-explanatory. Note that cm stands for confusion matrix here.
* Src: the "code" part of the repo. This has scripts for the LLMaAA analysis, as well as for more supportive tasks. 

Note: the first author also keeps an archive with files deleted from this folder. If you feel something is missing, you can contact him (see paper for contact details).
