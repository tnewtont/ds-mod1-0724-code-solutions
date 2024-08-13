
#1 In the ‘dsstudent’ database, create a permanent table named “customer_{your_name}.”
USE dsstudent;

CREATE TABLE customer_newton
	(customer_id smallint,
	name varchar(20),
	location varchar(20),
	total_expenditure varchar(20),
	CONSTRAINT pk_customer_newton PRIMARY KEY (customer_id)
	);

#2 Insert the following records to the “customer_{your_name}” table:

INSERT INTO customer_newton (customer_id, name, location, total_expenditure)
VALUES (1701, 'John', 'Newport Beach, CA', '2000'),
		(1707, 'Tracy', 'Irvine, CA', '1500'),
		(1711, 'Daniel', 'Newport Beach, CA', '2500'),
		(1703, 'Ella', 'Santa Ana, CA', '1800'),
		(1708, 'Mel', 'Orange, CA', '1700'),
		(1716, 'Steve', 'Irvine, CA', '18000');
		
#3 Oops! The value in the field ”total_expenditure” of Steve is not correct. It should be “1800.” Can you update this record?
UPDATE customer_newton
SET total_expenditure = '1800'
WHERE customer_id = 1716;


# 4 We would like to update our customer data. Can you insert a new column called “gender” in the “customer_{your_name}” table?
ALTER TABLE customer_newton
ADD gender varchar(20);


# 5 Then, update the field “gender” with the following records:
# Set all to M initially
UPDATE customer_newton
SET gender = 'M';

# Correct some to F
UPDATE customer_newton
SET gender = 'F'
WHERE customer_id in (1703, 1707, 1708);

#6 The customer, Steve, decides to quit our membership program, so delete his record from the “customer_{your_name}” table.
DELETE FROM customer_newton
WHERE customer_id = 1716;

#7 Add a new column called “store” in the table “customer_{your_name}”
ALTER TABLE customer_newton
ADD store varchar(20);

#8 Then, delete the column called “store” in the table “customer_{your_name}” because you accidentally added it.
ALTER TABLE customer_newton
DROP COLUMN store;

#9 Use “SELECT” & “FROM” to query the whole table “customer_{your_name}”
SELECT *
FROM customer_newton;

#10 Return “name” and “total_expenditure” fields from the table “customer_{your_name}”
SELECT name, total_expenditure
FROM customer_newton;

#11 Return “name” and “total_expenditure” fields from the table “customer_{your_name}” by using column alias (“AS” keyword)

SELECT name AS n, total_expenditure AS total_exp
FROM customer_newton;

#12 Change the datatype of the field “total_expenditure” from “VARCHAR” to ”SMALLINT”
ALTER TABLE customer_newton
MODIFY COLUMN total_expenditure smallint;

#13 Sort the field “total_expenditure” in descending order
SELECT total_expenditure
FROM customer_newton
ORDER BY 1 DESC;

#14 Return the top 3 customer names with the highest expenditure amount from the table “customer_{your_name}”
SELECT name, total_expenditure
FROM customer_newton
ORDER BY 2 DESC
LIMIT 3;

#15 Return the number of unique values of the field “location” and use the column alias to name the return field as “nuniques”
SELECT COUNT(DISTINCT(location)) AS nuniques
FROM customer_newton;

#16 Return the unique values of the field “location” and use the column alias to name the return field as “unique_cities”
SELECT DISTINCT(location) AS unique_cities
FROM customer_newton;

#17 Return the data where the gender is male.
SELECT *
FROM customer_newton
WHERE gender = 'M';

#18 Return the data where the gender is female.
SELECT *
FROM customer_newton
WHERE gender = 'F';

#19 Return the data where the location is “Irvine, CA”
SELECT *
FROM customer_newton
WHERE location = 'Irvine, CA'

#20 Return “name” and “location” where the ”total_expenditure” is less than 2000 and sort the result by the field “name” in ascending order.
SELECT name, location
FROM customer_newton
WHERE total_expenditure < 2000
ORDER BY 1;

#21 Drop the table “customer_{your_name}” after you finish all the questions.
DROP TABLE customer_newton;


