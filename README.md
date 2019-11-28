# Analyze A/B Test Result

A company has developed a new web page in order to try and increase the number of users who "convert," meaning the number of users who decide to pay for the company's product. Your goal is to help the company understand if they should implement this new page, keep the old page, or perhaps run the experiment longer to make their decision.

# Installations

You will need an installation of Python, plus the following libraries:

  - [Anaconda](https://www.anaconda.com/distribution/)
  - NumPy 
```python
pip install numpy
```
  - Pandas
```python
pip install pandas
```
- Matplotlib
 ```python
pip install matplotlib
```   
- Statsmodels
 ```python
pip install statsmodels
```   
In case that you don´t have the package intaller __pip__, you can follow this link:
[pip - The Python Package Installer](https://pip.pypa.io/en/stable/)

# Routines
- NumPy Random Sampling
  - [numpy.random.choice](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.choice.html#numpy.random.choice)
  - [numpy.random.normal](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html#numpy.random.normal)

# Concepts

- **Statistical Concepts**
  - [Basic Terminology](https://newonlinecourses.science.psu.edu/statprogram/reviews/statistical-concepts/terminology)
  - [Confidence Intervals](https://newonlinecourses.science.psu.edu/statprogram/reviews/statistical-concepts/confidence-intervals)
  - [Hypothesis Testing](https://newonlinecourses.science.psu.edu/statprogram/reviews/statistical-concepts/hypothesis-testing)
  - [Sampling Distribution](https://en.wikipedia.org/wiki/Sampling_distribution)
  - [Normal Distribution](https://en.wikipedia.org/wiki/Normal_distribution)
  
- **P-Value** 

The probability of obtaining the observed statistic or a "more extreme" value (by extreme we just mean more in favour of the alternate hypothesis mean) if the null hypothesis is true

[What is a p_value?](https://rebeccaebarnes.github.io/2018/05/01/what-is-a-p-value)

- **Logist Regression**

In statistics, the logistic model is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick. This can be extended to model several classes of events such as determining whether an image contains a cat, dog, lion, etc... Each object being detected in the image would be assigned a probability between 0 and 1 and the sum adding to one.

[Logist Regression](https://en.wikipedia.org/wiki/Logistic_regression)

# Libraries

- [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)

It's a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
