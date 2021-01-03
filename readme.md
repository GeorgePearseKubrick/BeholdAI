# Summary of Approach

First I remove any data that is of questionable quality, there are two types of duplicates in these datasets (model_output 
and radiologist_labels) 'true' duplicates e.g. rows which are identical across all columns and partial duplicates e.g.
the identifier (accession number) is the same but values in the columns are different. In the first case a single row 
can be kept, in the latter case I've selected to remove all rows with that accession number. They could instead 
all be left in and be considered independent indicators of the algorithms quality.

Having viewed the data I considered it important to return both the median and mean time taken to process an exam 
because the distribution (long tail) means there's a large disparity between the two measures.

I selected the f-beta score, it is impeded by the imbalance of the dataset but it at least properly takes into consideration
the weighting of priorities / implications of different biases. I considered using Matthews correlation coefficient
which considers all quadrants of the confusion matrix (this article was convincing https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7)
but didn't think it sufficiently includes a consideration of cost / real-life implications. 

I have left some arguably unnecessary data in, I figured that any data that was necessary to sense-check the metrics/results 
is therefore useful in the tool itself. The solution is designed to also serve as a check on any problems with the upstream 
data pipeline and highlight any alarm bells (e.g. surge in duplicates).

## Running instructions

Assuming you've already run pip install virtualenv, open up powershell/git bash (therefore should be the same on linux), navigate to the repo.
```
virtualenv beholdenv
. beholdenv/Scripts/activate
pip install requirements.txt
streamlit run BeholdAI.py
```