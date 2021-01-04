# Summary of Approach


First I remove any data that is of questionable quality, there are two types of duplicates in these datasets (model_output 
and radiologist_labels) 'true' duplicates e.g. rows which are identical across all columns and partial duplicates e.g.
the identifier (accession number) is the same but values in the columns are different. In the first case a single row 
can be kept, in the latter case I've selected to remove all rows with that accession number.

I was least sure about what to do with rows that had duplicate accession_numbers in model_outputs but different values in 
the 'Normal' column. Leaving them would leave a bias in the metric towards whatever category of X-ray requires multiple 
runs (probably difficult cases) but taking them out removed ~ 2500 rows. Averaging over them would be almost like assessing 
a different model.

I selected the f-beta score for the classification metric because it can be tuned to the implications of different biases,
but I've still stated what I suspect is a reasonable range. I considered using Matthews correlation coefficient because 
it incorporates all quadrants of the confusion matrix but didn't think it sufficiently includes a consideration of
implications.

I figured that any data that was necessary to sense-check the metrics/results is therefore useful in the tool itself. 
The solution is designed to also serve as a check on any problems with the upstream data pipeline and highlight any 
alarm bells (e.g. surge in duplicates). Internal products should have debugging built-in.

Dashboard can be viewed directly at https://share.streamlit.io/georgepearsekubrick/beholdai/main/BeholdAI.py if the streamlit server is still running.
