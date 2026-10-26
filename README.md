# Smartwatch Market Segmentation

## Project overview

This project analyses smartwatch consumer data to identify distinct market segments and evaluate their potential attractiveness from a marketing perspective.

The analysis combines unsupervised machine learning, statistical testing, segment profiling, and business-oriented evaluation to turn consumer preference data into potential marketing recommendations.

## Business questions

- How many distinct consumer segments can be identified in the sample?
- How do the segments differ in their smartwatch needs and preferences?
- Which segments appear most attractive for marketing purposes?
- How might the segments differ in suitability for potential business partners?
- What can willingness-to-pay analysis contribute to market evaluation?

## Methodology

### 1. Data preparation and exploration
- Load and inspect the segmentation and descriptor data.
- Review descriptive statistics and potential outliers.
- Prepare consumer rating variables for clustering.

### 2. Clustering and model evaluation
- Apply K-Means clustering and Ward hierarchical clustering.
- Compare candidate solutions using silhouette score, Calinski-Harabasz score, and Davies-Bouldin score.
- Examine bootstrap and split-half stability.
- Compare clustering solutions using the Adjusted Rand Index.

### 3. Segment profiling
- Compare consumer needs and preferences across the identified segments.
- Use descriptive summaries and appropriate statistical tests to assess differences.
- Examine categorical descriptors and segment characteristics.

### 4. Business evaluation
- Assess segment attractiveness using a weighted scoring framework.
- Conduct sensitivity analysis to examine how scores change under alternative weighting assumptions.
- Explore potential partner fit and willingness-to-pay patterns.

## Key findings

Replace this section with findings from the actual model outputs.

- **Number of segments:** [Insert the selected number and explain why.]
- **Segment profiles:** [Describe each segment using its observed characteristics.]
- **Most attractive segment:** [Insert the result and supporting evidence.]
- **Partner fit:** [Summarise the actual partner-fit results.]
- **Willingness to pay:** [Summarise the observed pattern and its business implications.]

## Tools and technologies

- Python
- pandas and NumPy
- scikit-learn
- SciPy
- Matplotlib and Seaborn
- Excel data processing

## Limitations

The findings describe the available respondent sample and should not automatically be generalised to the entire smartwatch market.

Cluster selection and segment attractiveness depend on methodological choices, data quality, and scoring assumptions. The attractiveness scores are decision-support estimates, not objective measures of market success.

## How to run

1. Install the dependencies listed in `requirements.txt`.
2. Place an authorised copy of the required workbook in the expected data folder.
3. Confirm that the input filenames and sheet names match the script.
4. Run:

   `python smartwatch_analysis.py`

5. Review the generated outputs.

## Author

Balaji Mallik

MSc Consumer Analytics and Marketing Strategy  
University of Leeds

