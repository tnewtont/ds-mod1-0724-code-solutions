#1 Create a temp table to store the row quantity of each table in “loandb” and the temp table includes 2 columns,
# one is “table_name” and the other is “row_quantity.” Show the table in the end. After take a screenshot of the
# result, then, drop the table.

USE loandb;

CREATE TEMPORARY TABLE dsstudent.quantities
	(table_name varchar(30),
	row_quantity int);
	
# Is there a more succinct way of doing this without using several COUNT(*) queries?

SELECT COUNT(*)
FROM train; #307511

SELECT COUNT(*)
FROM bureau; #1716428

SELECT COUNT(*)
FROM bureau_balance; #27299925

SELECT COUNT(*)
FROM previous_application; #1670214

SELECT COUNT(*)
FROM installments_payments; #13605401

SELECT COUNT(*)
FROM POS_CASH_balance; #10001358

SELECT COUNT(*)
FROM credit_card_balance; #3840312

INSERT INTO dsstudent.quantities (table_name, row_quantity)
VALUES ('train', 307511),
		('bureau', 1716428),
		('bureau_balance', 27299925),
		('previous_application', 1670214),
		('installments_payments', 13605401),
		('POS_CASH_balance', 10001358),
		('credit_card_balance', 3840312);

#2 Show the monthly and annual income

SELECT AMT_INCOME_TOTAL AS annual_income, (AMT_INCOME_TOTAL/12) AS monthly_income
FROM loandb.train;

#3 Transform the “DAYS_BIRTH” column by dividing “-365” and round the value to the integer place. Call this column as “age.”

SELECT ROUND(DAYS_BIRTH/-365) AS age
FROM loandb.train;

#4 Show the quantity of each occupation type and sort the quantity in descending order.
SELECT OCCUPATION_TYPE AS occupation_type, COUNT(OCCUPATION_TYPE) AS quantity
FROM loandb.train
GROUP BY 1
HAVING occupation_type IS NOT NULL
ORDER BY 2 DESC;

#5 In the field “DAYS_EMPLOYED”, the maximum value in this field is bad data, can you write a conditional logic to
#mark these bad data as “bad data”, and other values are “normal data” in a new field called “Flag_for_bad_data”?

# Is there a way to incorporate MAX into the CASE statement?

SELECT MAX(DAYS_EMPLOYED)
FROM loandb.train;

SELECT DAYS_EMPLOYED,
	CASE
		WHEN DAYS_EMPLOYED = (SELECT MAX(DAYS_EMPLOYED) FROM loandb.train) THEN 'bad data'
		ELSE 'normal data'
	END Flag_for_bad_data
FROM loandb.train;

#6 Can you show the minimum and maximum values for both “DAYS_INSTALLMENT” & “DAYS_ENTRY_PAYMENT”
# fields in the “installment_payments” table for default v.s. non-default groups of clients?


USE loandb;


# This ran successfully after 41 minutes....
# I realized the problem was in the SELECT statement where I specified the table before each column, and then when I
# troubleshoot the query bit-by-bit, I found out it immediately broke down on the first INNER JOIN.
# That made me realize it was probably better to just state the column names without appending any specific table beforehand.

CREATE TEMPORARY TABLE dsstudent.bigmerge
SELECT TARGET, DAYS_INSTALMENT, DAYS_ENTRY_PAYMENT
FROM loandb.installments_payments AS ip INNER JOIN loandb.credit_card_balance AS ccb ON ip.SK_ID_PREV = ccb.SK_ID_PREV
				INNER JOIN loandb.previous_application AS pa ON ccb.SK_ID_PREV = pa.SK_ID_PREV				
				INNER JOIN loandb.train AS t ON pa.SK_ID_CURR = t.SK_ID_CURR;
			
CREATE TEMPORARY TABLE dsstudent.bigmerge
SELECT t.TARGET, ip.DAYS_INSTALMENT, ip.DAYS_ENTRY_PAYMENT
FROM loandb.installments_payments AS ip INNER JOIN loandb.credit_card_balance AS ccb ON ip.SK_ID_PREV = ccb.SK_ID_PREV
				INNER JOIN loandb.previous_application AS pa ON ccb.SK_ID_PREV = pa.SK_ID_PREV				
				INNER JOIN loandb.train AS t ON pa.SK_ID_CURR = t.SK_ID_CURR;
			
			
							
SELECT TARGET, MIN(DAYS_INSTALMENT) AS min_day_installment, MAX(DAYS_INSTALMENT) AS max_day_installment, MIN(DAYS_ENTRY_PAYMENT) AS min_days_entry_payment, MAX(DAYS_ENTRY_PAYMENT) AS max_days_entry_payment
FROM dsstudent.bigmerge
GROUP BY TARGET
HAVING TARGET IS NOT NULL
ORDER BY TARGET ASC;

