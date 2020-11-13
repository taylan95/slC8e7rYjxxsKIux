# Term Deposit Prediction

## Context&Goals

 Term deposits are usually short-term deposits with maturities ranging from one month to a few years. 
 The customer must understand when buying a term deposit that they can withdraw their funds only after the term ends.
 We will predict whether the customer will subscribe (yes / no) to the term deposit (variable y).
 
## Content

* age : age of customer (numeric)
* job : type of job (categorical)
* marital : marital status (categorical)
* education (categorical)
* default: has credit in default? (binary)
* balance: average yearly balance, in euros (numeric)
* housing: has a housing loan? (binary)
* loan: has personal loan? (binary)
* contact: contact communication type (categorical)
* day: last contact day of the month (numeric)
* month: last contact month of year (categorical)
* duration: last contact duration, in seconds (numeric)
* campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
* y: has the client subscribed to a term deposit? (binary)

1. Data & Data overview
2. Data Manipulation, Exploratory Data Analysis
3. Data preprocessing, Feature Selection 
4. Model Building
5. Model Performances
6. Segmentation (Bonus)
7. Summary and Conclusion

[linkedin]: https://www.linkedin.com/in/taylan-polat/

 You can download requirements via "pip install -r requirements"

## Summary&Conclusion

* It was observed that people who were specified as 'yes' in the default variable did not have a term deposit status or when the Loan variable was examined,
  those who were specified as 'yes' did not buy term deposits. 
* In these variables, the purchasing rate of the people indicated as 'no' is higher. There may be more focus for these people in future marketing strategies. 
* When we examine the product sales channels, it is concluded that talking by phone is not effective, those made with other options are relatively higher than the phone, 
  and the calls made with cellular are more effective in selling the product.It has been observed that product sales increased mostly in May and in the last days of the months. 
  It is observed that product sales in the winter season decrease on customer basis.
* It has been observed that people with secondary education profiles prefer more in purchasing products. 
  As the most preferred group to buy in the age group is between the ages of 20-30, this situation shows more in the student group in the Job portfolio.
* It has been observed that those who are determined as 'yes' for the housing variable can buy the product relatively easily, while those who are above average for the balance variable prefer to buy more.
* When the 4 segments created by applying KMeans are examined, when we compare the percentages of whether the customers in these segments buy the product or not, 
  it is observed that the segment with the label "1", also known as the "best segment", has the biggest share in purchasing the product.










