# Natural Language Generation
A collection of recent progress on Natural Language Generation, including papers, codes

- [Natural Language Generation](#natural-language-generation)
   - [Neural Data-to-Text Generation](#neural-data-to-text-generation)
   - [Traditional Data-to-Text Generation](#traditional-data-to-text-generation)
   - [Neural Normal Text Generation](#neural-normal-text-generation)
   - [Probabilistic & Machine Learning based Method for Normal Text Generation](#probabilistic-machine-learning-based-method-for-text-generation)
   - [Probabilistic Method for Generation](#probabilistic-method-for-generation)
   - [Text Generation for Time-Series Data](#text-generation-for-time-series-data)
   - [Text Generation with External Knowledge](#text-generation-with-externel-knowledge)
   - [General Framework for Text Generation](#general-framework-for-text-generation)
   - [DataSet](#dataset)
   - [Researchers](#researchers)
   - [Demo](#demo)
   - [Conferences](#conferences)
   
## Neural Data-to-Text Generation

__Table-to-Text Generation with Effective Hierarchical Encoder on Three Dimensions (Row, Column and Time)__. EMNLP 2019. __Heng Gong, Xiaocheng Feng__. [[Paper](https://www.aclweb.org/anthology/D19-1310.pdf)] [[Code](https://github.com/ernestgong/data2text-three-dimensions)]

__Enhancing Neural Data-To-Text Generation Models with External Background Knowledge__. EMNLP 2019. __Shuang Chen, Jinpeng Wang__. [[Paper](https://www.aclweb.org/anthology/D19-1299.pdf)]

__GTR-LSTM:A Triple Encoder for Sentence Generation from RDF Data__. ACL 2018. [[Paper](https://www.aclweb.org/anthology/P18-1151.pdf)]

__Neural data-to-text generation: A comparison between pipeline and end-to-end architectures__. EMNLP 2019. __Thiago Castro Ferreira, Chris van der Lee__. [[Paper](https://www.aclweb.org/anthology/D19-1052.pdf)] [[Code](https://github.com/ThiagoCF05/DeepNLG)]

__Enhanced Transformer Model for Data-to-Text Generation__. WNGT 2019. __Li Gong, Josep Crego, Jean Senellart__. [[Paper](www.aclweb.org/anthology/D19-56)]

__Data-to-Text Generation with Entity Modeling__. ACL 2019. __Ratish Puduppully, Li Dong, and Mirella Lapata__. [[Paper](https://www.aclweb.org/anthology/P19-1195.pdf)] [[Code](https://github.com/ratishsp/data2text-entity-py)]

__Data-to-Text Generation with Content Selection and Planning__. AAAI 2019. __Ratish Puduppully, Li Dong, and Mirella Lapata__. [[Paper](https://arxiv.org/pdf/1809.00582v1.pdf)] [[Code](https://github.com/ratishsp/data2text-plan-py)]

__Learning to Select, Track, and Generate for Data-to-Text__. ACL 2019. __Hayate Iso, Yui Uehara, Tatsuya Ishigaki__. [[Paper](https://arxiv.org/pdf/1907.09699.pdf)] [[Code](https://github.com/aistairc/sports-reporter)] [[Data](https://github.com/aistairc/rotowire-modified)]

__Bootstrapping Generators from Noisy Data__. NAACL 2019. __Laura Perez-Beltrachini and Mirella Lapata__. [[Paper](https://arxiv.org/pdf/1804.06385.pdf)] [[Code](https://github.com/EdinburghNLP/wikigen)]

__Table-to-Text Generation by Structure-aware Seq2Seq Learning__. AAAI 2018. __Tianyu Liu, Kexiang Wang, Lei Sha__. [[Paper](https://arxiv.org/pdf/1711.09724.pdf)] [[Code](https://github.com/tyliupku/wiki2bio)]

__Table-to-Text: Describing Table Region with Natural Language__. AAAI 2018. __Junwei Bao, Duyu Tang, Nan Duan__. [[Paper](https://arxiv.org/pdf/1805.11234v1.pdf)] [[Code]()]

__A Mixed Hierarchical Attention based Encoder-Decoder Approach for Standard Table Summarization__. NAACL 2018. __Parag Jain Anirban Laha__. [[Paper](https://arxiv.org/pdf/1804.07790.pdf)] 

__Order-Planning Neural Text Generation From Structured Data__. CoRR 2017. __Lei Sha__. [[Paper](https://arxiv.org/pdf/1709.00155.pdf)] 

__Learning to generate one-sentence biographies from Wikidata__. ECACL 2017. __Andrew Chisholm, Will Radford__. [[Paper](https://www.aclweb.org/anthology/E17-1060.pdf)] [[Code](https://github.com/andychisholm/mimo)]

__What to talk about and how? Selective Generation using LSTMs with Coarse-to-Fine Alignment__. NAACL 2016. __Hongyuan Mei, Mohit Bansal__. [[Paper](https://arxiv.org/pdf/1509.00838.pdf)]



## Traditional Data-to-Text Generation
__Probabilistic Verb Selection for Data-to-Text Generation__. ACL 2018. __Dell Zhang, Jiahao Yuan__. [[Paper](https://www.aclweb.org/anthology/Q18-1038.pdf)]

__Learning Latent Semantic Annotations for Grounding Natural Language to Structured Data__. EMNLP 2018. __Guanghui Qin, Jin-Ge Yao, Xuening Wang, Jinpeng Wang, Chin-Yew Lin__. [[Paper](https://www.aclweb.org/anthology/D18-1411.pdf)] [[Code](https://github.com/hiaoxui/D2T-Grounding)]

__A Statistical Framework for Product Description Generation__. IJCNLP 2017. __Jinpeng Wang, Yutai Hou__. [[Paper](https://www.aclweb.org/anthology/I17-2032.pdf)]

__Unsupervised Concept-to-Text Generation with Hypergraphs__. NAACL 2012. __Ioannis Konstas and Mirella Lapata__. [[Paper](http://aclweb.org/anthology/N12-1093)]

__Inducing Document Plans for Concept-to-Text Generation__. EMNLP 2013. __Ioannis Konstas and Mirella Lapata__. [[Paper](http://aclweb.org/anthology/D13-1157)]

__An Architecture for Data-to-Text Systems__. WNLG 2007. __Ehud Reiter__. [[Paper](http://dl.acm.org/ft_gateway.cfm?id=1610180&type=pdf)]

__Collective Content Selection for Concept-to-Text Generation__. EMNLP 2005. __Regina Barzilay and Mirella Lapata__. [[Paper](http://aclweb.org/anthology/H05-1042)]

__Statistical Acquisition of Content Selection Rules for Natural Language Generation__. EMNLP 2003. __Pablo A Duboue and Kathleen R McKeown__. [[Paper](https://www.aclweb.org/anthology/W03-1016.pdf)]

__Empirically Estimating Order Constraints for Content Planning in Generation__. ACL 2001. __Pablo A Duboue and Kathleen R McKeown__. [[Paper](https://www.aclweb.org/anthology/P01-1023.pdf)]

## Neural Normal Text Generation 
__NeuralREG: An End-to-End Approach to Referring Expression Generation__. ACL 2018. __Thiago Castro Ferreira,Diego Moussallem__. [[Paper](https://www.aclweb.org/anthology/P18-1182.pdf)] [[Code](https://github.com/ThiagoCF05/NeuralREG)]

## Probabilistic & Machine Learning based Method for Normal Text Generation
__A Simple Domain-Independent Probabilistic Approach to Generation__. EMNLP 2010. __Gabor Angeli, Percy Liang, and Dan Klein__. [[Paper](http://aclweb.org/anthology/D10-1049)]

__Probabilistic Generation of Weather Forecast Texts__. NAACL 2007. __Anja Belz__. [[Paper](http://aclweb.org/anthology/N07-1021)]

__Natural Language Generation with Tree Conditional Random Fields__. EMNLP 2019. __Wei Lu, Hwee Tou Ng, and Wee Sun Lee__. [[Paper](http://aclweb.org/anthology/D09-1042)]

## Text Generation for Time-Series Data
__Generating Market Comment Referring to External Resources__. INLG 2018. __Tatsuya Aoki, Akira Miyazawa__. [[Paper](https://www.aclweb.org/anthology/W18-6515)] [[Code](https://github.com/aistairc/market-reporter)]

__Learning to Generate Market Comments from Stocks Prices__. ACL 2017. __Soichiro Murakami, Akihiko Watanabe__. [[Paper](https://doi.org/10.18653/v1/P17-1126)]

## Text Generation with External Knowledge

## General Framework for Text Generation
__Texar: A Modularized, Versatile, and Extensible Toolkit for Text Generation__. ACL 2019 demo. [[Paper](https://arxiv.org/pdf/1809.00794.pdf)] [[Code](https://github.com/asyml/texar)]

__Data2Text Studio: Automated Text Generation from Structrured Data__. EMNLP 2018. [[Paper](https://pdfs.semanticscholar.org/79dd/2ee41e4a7de3b3142fea43b8c48d20224ef2.pdf)]

## DataSet
__WikiInfo2Text:Enhancing Neural Data-To-Text Generation Models with External Background Knowledge__. EMNLP, 2019. __Shuang Chen, Jinpeng Wang__. [[Paper](https://www.aclweb.org/anthology/D19-1299.pdf)] [[Data](https://github.com/hitercs/WikiInfo2Text)] 

__ROTOWIRE: Challenges in Data-to-Document Generation__. EMNLP 2017. __Sam Wiseman, Stuart M. Shieber, Alexander M. Rush__. [[Paper](https://arxiv.org/pdf/1707.08052v1.pdf)] [[Data](https://github.com/harvardnlp/boxscore-data)]

__WebNLG: The WebNLG Challenge:  Generating text from RDF data__. INLG 2017. __Claire Gardent, Anastasia Shimorina, Shashi Narayan, and Laura Perez-Beltrachini__. [[Paper](https://www.aclweb.org/anthology/W17-3518.pdf)] [[Data](https://www.aclweb.org/portal/content/webnlg-challenge-first-call-participation-0)]

__WikiBio: Neural Text Generation from Structured Data with Application to the Biography Domain__. EMNLP 2016. __Rémi Lebret, David Grangier and Michael Auli__. [[Paper](http://arxiv.org/abs/1603.07771)] [[Data](https://github.com/DavidGrangier/wikipedia-biography-dataset)]

__Weather: Learning Semantic Correspondences with less supervision__. ACL 2009. __Liang, Percy, Michael I. Jordan, and Dan Klein__. [[Paper](https://pdfs.semanticscholar.org/c53e/9da715fb48489939ad832d2db91e022d3ff5.pdf?_ga=2.144448496.477113533.1575267799-892594366.1561096874)] [[Data](https://link.zhihu.com/?target=https%3A//cs.stanford.edu/~pliang/data/weather-data.zip)]

__SportsCasting: Learning to Sportscast: A test of grounded language acquisition__. ICML 2008. __Chen, David L., and Raymond J. Mooney__. [[Paper](http://machinelearning.org/archive/icml2008/papers/304.pdf)] [[Data](https://link.zhihu.com/?target=http%3A//www.cs.utexas.edu/~ml/clamp/sportscasting/data.tar.gz)]

## Researchers

## Demo

## Conferences
   __ACL__. [WebSite]
   
   __NAACL__. [WebSite]
   
   __EMNLP__. [WebSite]
   
   __INLG__. [WebSite]

   __WNGT__. [WebSite]
