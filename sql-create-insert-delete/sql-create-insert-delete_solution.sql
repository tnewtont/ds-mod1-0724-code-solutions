USE dsstudent;

#1 Create table
CREATE TABLE person_newton
	(person_id SMALLINT, first_name VARCHAR(30), last_name VARCHAR(30), city VARCHAR(30),
	CONSTRAINT pk_person_newton PRIMARY KEY (person_id));
	
#2 Insert a row of data
INSERT INTO person_newton (person_id, first_name, last_name, city)
VALUES (1, 'Newton', 'Tran', 'Newport Beach');

#3 Insert two rows of data
INSERT INTO person_newton (person_id, first_name, last_name, city)
VALUES (2, 'Truc', 'Tran', 'Newport Beach');

INSERT INTO person_newton (person_id, first_name, last_name, city)
VALUES (3, 'Jim', 'Nguyen', 'Milpitas');
# COLUMN IN (list)

#4 Add column called 'gender'
ALTER TABLE person_newton
ADD gender VARCHAR(8);

#5 Update 'gender' for each row
UPDATE person_newton
SET gender = 'male'
WHERE person_id = 1;

UPDATE person_newton
SET gender = 'male'
WHERE person_id = 2;

UPDATE person_newton
SET gender = 'male'
WHERE person_id = 3;

# Deleting and dropping data
#6 delete 'gender'
ALTER TABLE person_newton 
DROP COLUMN gender;

#7 Delete row where person_id = 2
DELETE FROM person_newton 
WHERE person_id = 2;

#8 Drop the person_newton table
DROP TABLE person_newton;
