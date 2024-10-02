---
title: "1. BASICS OF PYTHON"
layout: default
nav_order: 3
---
# **The Basics of Python**
This interactive tutorial will get you started on your data-exploration road and make you familiar with some core
concepts of Python programming and data analysis.  

> **Notice:**  
> In all following sections I will insert some code snippets. You are very much encouraged to copy and paste them with the button on the top right
and run them in your IDE (e.g. Spyder).

### Table of Contents
1. [General stuff about Python](#1-general-stuff-about-python)
2. [Data Types and Variables](#2-data-types-and-variables)
3. [Operators](#3-operators)
4. [Loops and Conditionals](#4-loops-and-conditionals)
5. [Functions and Classes](#5-functions-and-classes)

Notice that it is not at all expected that you learn all these things and they are burnt into your brain (!!!!!). 
It is more of a broad intrdocution to all the basics so you have hard of them, but programmers do look up stuff
all the time! So don't worry if it is a lot of input right now, just try to understand the concepts and you 
can always come back and find help in here, in the internet or from me directly.
  
Here are some useful ressources to look things up:

[Link: w3schools.com: Tutorials on many topics where you can quickly look up things...](https://www.w3schools.com/python/python_intro.asp){:target="_blank"}{:rel="noopener noreferrer"}  
[Link: geeks4geeks.com: Another nice overview of many functionalities of Python (requires login)...](https://www.geeksforgeeks.org/python-cheat-sheet/){:target="_blank"}{:rel="noopener noreferrer"}   
Geeksforgeeks requires you to make an account or use e.g. a google login, but it features many tutorials, project ideas, quizzes and so on on many programming languages and general
topics such as Machine Learning, Data Visualization, Data Science, Web Development and many more  
[Link: Pandas cheat sheet: Later on we will use the library "Pandas" (so cute!) for data handling. A nice cheat sheet is provided by the developers...](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf){:target="_blank"}{:rel="noopener noreferrer"}   


## 1. General things about Python
I quickly want to highlight some things which are special about Python compared to many other programming languages and that one needs to get used to.  

### Python is indentation sensitive
In Python it matters, how far you indent your lines, meaning how much space you have at the beginning of a line.
As an example this will work:
```python
a = 5
b = 1
```
but this will throw an error:
```python
a = 5
  b = 5
```
will result in an error:
```python
File "<stdin>", line 1
  b = 5
IndentationError: unexpected indent
```

### Variables
Generally in Python variables are created by assigning a value to them with an equal sign, just like we did above. Theire output can be shown by just typing the variable:
```python
a = 5
a
5
``` 

### Comments
Comments are lines in the code that are not executed and are there for documentation. For now it is a good idea to use comments in your code to keep track of what is happening where.  
Single line comments are always created with an '#'. Everything after that symbol in the line is not executed. Multi-line comments can be written by enclosing them in three ':
```python
# first I create a single line comment, this is not executed
a = 5 # this line is executed, but the comment gets ignored
''' 
Now I write a multi-line comment
I can continue the comment on the next line
b = 5 <-- this is ignored
'''
```
### Python is 0-indexed
In Python, the first index in for example a list always has the number 0! This takes some time to get used to, especially if you come from e.g. R which has 1-based indexing, but most programming lanuge handle indexing like that and it is worth getting used to it. I won't go into why it is handled like that but there are many discussions on the internet about it, feel free to dive in if you feel like diving into a
rabbit-hole ;)


### Separators
In Python separators for decimal numbers are ALWAYS dots! Commas are used e.g. to separete variables from each other or entries in a list
```python
correct_decimal = 2.5
correct_decimal
2.5
```

### Naming of variables, functions, and anything at all
This is not Python-specific but a very important note!  
**Always use descriptive names for variables, functions or anything that you give a name!**  
Especially in scientific programming you see it time and time again that people name variables and functions
using abbreviations that just came to their mind. This makes code much, **much** harder to read and to use by other people or your own future self.
It happens so often that people look back at what they wrote 3 weeks ago and do not understand half of it because they did not give descriptive names.  

You can also use comments to document your code a bit, but that always takes up extra space, often does not look good because you barely keep the same
formatting throughout the code and gives the next user more work to do when trying to understand the code. Just making the code explain itself is the best solution of all.  
Here is a very simple example:  
```python
# Bad code with abbreviations  
# it requires the user to interprete the variables and look at 
# used functions to understand what this even does
l = [1,5,12,17,18,14,11]
n = len(l)
s = sum(l)
m = s/n

# Fixing it with comments
# With comments we require the user to read all the extra text to 
# 1) understand what the data is
# 2) understand what is calculated

l = [1,5,12,17,18,14,11] # a list of temperature values
n = len(l)  # get total number of samples
s = sum(l)  # get total sum of samples
m = s/n     # calculate the mean value


# This gets so much easier to read when using declarative naming.
# You dont even have to look into the function to understand what it is doing:
monthly_temperature = [1,5,12,17,18,14,11]
number_of_samples = len(monthly_temperature) 
sum_of_samples = sum(monthly_temperature)
mean_monthly_temperature = sum_of_samples/number_of_samples 

# This does not mean that naming has to replace comments completely 
# (although some people argue like that). It is still alright to use comments
# to clarify parts of your code, just try keep it to a minimum and make the
# code as self-explanatory as possible!

```

## 2. Data Types and Variables
Python knows different types of data. A number is a different kind of variable than a word. That helps organizing the variables and defines, which operations are
possible with which data. For example, computing the mean of a word would be difficult, just as translating a number to all-uppercase letters...  

In Python you dont have to define the data type yourself because Python is smart and finds the type of data on its own. For example when we define a number  
Python will understand and give it the type "int" or "float", which means "integer" or "floating point number" (decimal)
We will not cover all data types as we probably won't need all of them for our purpose. However these ones are important:  
  
**Primitive Datatypes**  
Primitive datatypes are simple constructs that consists basically of one chunk of information, e.g. a number or a word:
- **int**: Integer, a number without a floating point
- **float**: Floating point number, a number with decimals
- **str**: A string of characters, e.g. letters, words and sentences
- **bool**: Boolean, a value that can only be True or False. This helps us make decisions in our code  
  
**Non-Primitive Datatypes**  
Non-primitive datatypes consist of aggregations of primitive datatypes. A list for example holds several numbers or words or something else  
- **list**: An ordered sequence of data, for example [1,2,3] is a list where each of the entries have a specific position and the entries can be accessed by indices
- **dict**: A non-ordered mapping that consists of keys and values. That simply means, we can not get entries from the dictionary by indices (e.g. the 0th entry in a dictionary) but instead grab data from the dictionary by using the key. Imagine it like a digital telephone-book. The comparison does not hold completely because in theory a telephone book is ordered, but you would never search the 5001231th entry in a telephone book. Instead you would search the phone number of  
Mr. Smith", so you go to the "key" Mr. Smith and get the "value" 0251/1234567.
  
Lets look at some examples for data types:  
```python
# Primitive datatypes:

letter_a = "a"                  # <-- a string 
name = "Josefine" # <-- a longer string 
age = 24                        # <-- an integer
total_playtime = 354.5          # <-- a float
is_injured = False              # <-- a boolean

# Non-Primitive datatypes:
# list: 
# a list is alwasys enclosed by brackets 
# and the items are separeted with commas:
scores_last_games = [5,3,0,1] 
# To access the values we can use the index, for example 
scores_last_games[0]  # <-- gets the first entry
scores_last_games[2]  # <-- gets the 3rd entry
scores_last_games[-1] # <-- gets the last entry
scores_last_games[-2] # <-- gets the second last entry

# dictionary:
# is always enclosed by {}, 
# and has the structure "key":value, lines are separeted by a comma.
josefine = {
  "age":age,
  "total_playtime":total_playtime,
  "is_injured":is_injured,
  "scores_last_games":scores_last_games
}
# Now the values of the dictionary can be accessed using the key like this:
josefine["age"]
24
# new entries can be added by assigning a value to a new key:
josefine["trikot_number"] = 9
```
You can always find the type of a variable by using the type() function (more on functions later):
```python
type(name)
type(age)
type(total_playtime)
type(is_injured)
type(scores_last_games)
type(josefine)
```

It is possible to change the type of a variable, but only if Python is able to understand what the outcome should be.  
The functions to do that have the same name as the target data type, for example int() or str():
```python
int("10")       # <-- this works
str(500)        # <-- this works
float(500)      #<-- this works
float("500.5")  #<-- this works
float("abc")    #<-- this won't work, how should you translate a word to a number?
```
One last thing is important to note. When you assign a non-primitive variable to another non-primitive variable, the two variables **share** the same data. That means,
when you manipulate one you also manipulate the other. This can lead to confusion when you don't keep it in mind.  
```python
list_1 = [1,2,3]  # a list is non-primitive
list_2 = list_1   # here we assign the non-primitive list_1 to the variable list_2
list_2.append(4)  # we add a fourth value, 4, to list_2
list_2            # list_2 is now [1,2,3,4]
list_1            # BUT! list_1 is now also [1,2,3,4]

# We can avoid this and extract the values from list_1 to create a completely new variable by using the .copy() function
list_1 = [1,2,3]
list_2 = list_1.copy()    # we copy the values of list_1 to the new variable list_2
list_2.append(4)          # we add a fourth value, 4, to the list list_2
list_2                    # list_2 is now [1,2,3,4]
list_1                    # list_1 is still [1,2,3]
```
On the other hand when you assign a variable containing a primitive datatype to another variable, the value gets simply
copied to the new variable. Here is an example:
```python
a = 5
b = a     # we assign the value of a to the variable b
b = b + 1 # we increase the value of b by one
b         # b is now 6
a         # a is still 5
```
## 3. Operators
An operator is something that allows you to interact with variables. Some very examples are mathematical operations or comparisons.  

### 3.1 Arithmetic operations
Most operations are very intuitive. For example you can add numbers and also add words to concatenate them, but you can not subtract words from each other...  
Here is a list of operations:
```python
a = 5
b = 10
word1 = "Hi"
word2 = "there"

# Airthmetic Operators:
c = a + b   # adding numbers
concatenated_words = word1 + " " + word2 # adding words
d = b - c   # subtracting numbers
e = a * b   # multiplying numbers
f = b / 5   # dividing numbers
g = a ** 2  # Exponentation, this is a²
h = 12 % 5  # Modulus, this returns the remaining amount after fitting one number into the other as many times as possible.
```

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}
<h3> Exercise </h3>
<p >With what you know so far, grab the scores josefine scored in the last games and compute the average amount of goals per game she scores</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
scores = josefine["scores_last_games"]
total_scores = scores[0] + scores[1] + scores[2] + scores[3]
mean_scores = total_scores / 4
```
<p >There are much better solutions to this, for example the iteration over all scores can be done with the built-in function sum()
and the total number of score-values can be found using the len() function. A one-line solution could look like this:</p>

```python
mean_scores = sum(josefine["scores_last_games"]) / len(josefine["scores_last_games"])
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>

### 3.2 Comparison operations
Comparison operations are used to compare values with each other in order to make decisions in your script.  
The output of a comparison is **always** a boolean value that is "True" if the comparison is evaluated as correct and "False" otherwise. 

```python
goals_team1 = 5
goals_team2 = 2

goals_team1 > goals_team2    # > larger than
goals_team1 >= goals_team2    # >= larger than or equal
goals_team1 < goals_team2    # < smaller than
goals_team1 <= goals_team2    # <= smaller than or equal
goals_team1 == goals_team2   # == equal
goals_team1 != goals_team2   # != not equal

# you can also store the result in a variable:
is_team1_winner = goals_team1 > goals_team2
is_team2_winner = goals_team1 < goals_team2
```

### 3.3 Logical operators
Logical operators can combine multiple comparisons. Namely there are three: **and**, **or** and **not**. The use of these is pretty intuitive.
If we combine two comparisons with an "and", the result is only True if all conditions hold.  
If we combine two comparisons with an "or", the result is True if one of the conditions hold, even if the other is False.
Not is a special case, that reverts the result. 

```python
# Lets use a new example
peter = {
  "age":24,
  "height":1.73,
  "is_enrolled": True
}
joana = {
  "age":25,
  "height":1.75,
  "is_enrolled": False
}
# Now we can do some comparisons:
is_peter_taller_and_older_than_joana = peter["age"] > joana["age"] and peter["height"] > joana["height"]
is_peter_not_enrolled = not peter["is_enrolled"]
is_joana_not_enrolled = not joana["is_enrolled"]
is_peter_or_joana_enrolled = peter["is_enrolled"] or joana["is_enrolled"]
```

### 3.4 Identity and membership operators
The identitiy operator "is" is to check whether two objects are the same. On the other side, the membership operator "in" checks whether an object is contained within another object.
Simple examples:
```python
a = [1,2,3] # a simple list
b = a       # we assign a to b, remember non-primitive data types?
a is b      # What will be the result of this?

1 in a      # we can test whether a contains a number 1
c = [a,b]   # here we create a new list that contains the lists a and b
a in b      # now we can check whether one of the lists is within another list
a in c
```

### Code block in details in a notice

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}

<h3> Exercise </h3>
<p >Now you know all about operators. Try to use your knowledge and figure out what we test for in the following operations and what the result is:</p>

{::options parse_block_html="true" /}

```python
joana = {
    "enrolled": True,
    "grade_ecophysiology": 1.3,
    "grade_archery": 1.3
}
alfonso = {
    "enrolled": True,
    "grade_ecophysiology": 1.7,
    "grade_archery": 4.3
}
legolas = {
    "enrolled": False,
    "grade_ecophysiology": 4.0,
    "grade_archery": 1.0
}

# 1.
a = (legolas["grade_ecophysiology"] < joana["grade_ecophysiology"]) and (legolas["grade_ecophysiology"] < alfonso["grade_ecophysiology"])
# 2.
b = (legolas["grade_archery"] < joana["grade_archery"]) and (legolas["grade_archery"] < alfonso["grade_archery"])
# 3.
c = not (legolas["grade_ecophysiology"] < joana ["grade_ecophysiology"]) or (alfonso["grade_ecophysiology"] < joana ["grade_ecophysiology"])
# 4.
d = joana["enrolled"] and alfonso["enrolled"] and legolas["enrolled"]
# 5.
e = alfonso["grade_ecophysiology"] > 4.0 or alfonso["grade_archery"] > 4.0
# 6.
f = (alfonso["grade_ecophysiology"] > 4.0 or alfonso["grade_archery"] > 4.0) or (legolas["grade_ecophysiology"] > 4.0 or legolas["grade_archery"] > 4.0) or (joana["grade_ecophysiology"] > 4.0 or joana["grade_archery"] > 4.0)

```

<details><summary markdown="span">Solution!</summary>
<ol>
<li>Check 1 tests whether legolas is the best ecophysiologist. The result is False.</li>
<li>Check 2 tests whether legolas is the best archer. The result is True.</li>
<li>Check 3 tests whether legolas or alfonso are better ecophysiologists than joana. With the "not" in the beginning, the result is turned into whether Joana is better than any of the two. The result is True.</li>
<li>Check 4 tests whether everyone is enrolled. The result is False. Legolas is probably buisy somewhere else...</li>
<li>Check 5 tests whether Alfonso failed one of the exams with a grade higher than 4.0. The result is True.</li>
<li>Check 6 tests whether anyone failed one of the exams with a grade higher than 4.0. The result is True.</li>
</ol>
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>


## 4. Conditionals and Loops
Conditionals and loops are constructs in your code that are often combined. A loop is used to do a certain task on many elements sequentially,
a conditional uses a certain condition (or truth-evaluation) to decide whether a piece of code should be executed.  

### 4.1 Conditionals
Remember how we talked about comparison, logical and identitiy and membership operators? They all result in a boolean, stating whether
a condition is True or False. We can make use of that by utilizing conditionals. Here is a simple example:
```python
is_peter_smart = True
if is_peter_smart == True:
    print("Peter is smart")
```
Notice how indentation plays a role here! We end the line of the if-check with a ":" and start the new line indented.
Indented lines signal a code block, that always belongs to the previous statement that ended with a ":".
  
In  the above example the print() command will be executed because the value of is_peter_smart is True.
If we check for a boolean value (True or False) we can also leave the comparison operation out
and ask very coloquially:
```python
is_peter_smart = True
if is_peter_smart:
    print("Peter is smart")
```
We can also define a code that should be executed, ONLY if the if-check is evaluated as False. For that 
we use the keyword "else". 
```python
is_peter_smart = True
if is_peter_smart:
    print("Peter is smart")
else:
    print("Peter is not smart")
```
Finally, you can also chain if-checks by using the "elif" keyword. This stands for "else if", meaning
that "if the previous checks failed and this check is evaluated as True, run the code"
```python
is_peter_smart = False
is_peter_big = True
if is_peter_smart:
    print("Peter is smart")
elif is_peter_big:
    print("Peter is big")
else:
    print("Peter is not smart and not big")
```

### 4.2 Loops
A loop is a structure that allows you to iteratively perform actions, either with several elements (e.g. stored in a list)
or while a specific condition holds. These two types are called "for-loops" and "while-loops". 
They always consist of two parts: The definition how and over what you want to iterate (or "loop") and 
the actual action you want to perform. 

#### 4.2.1 The for-loop
The most "classical" loop is the for-loop.  
The syntax is, as often in Python, held very simple. Here is an easy example:
```python
temperatures = [12,14,16,15,16,17,20,21]
for temperature in temperatures:
    print(temperature)
```
Notice that in the definition of the loop, we define a new variable called "temperature".
This variable represents the element we are currently working on in each step of the loop.
So in the first step, temperature is 12, in the next temperature is 14 and so on.  
There is one very handy built-in method that can give you both the value of the list-entry **and**
its corresponding index, called enumerate(). You can put them both in variables by using a comma
in the loop-definition. A quick demo:
```python
temperatures = [12,14,16,15,16,17,20,21]
hour_of_day = [8,9,10,11,12,13,14,15]

# When using enumerate, each iteration we get the index and value of the current list entry.
# So in the first loop index will be 0 and temperature 12, 
# next index will be 1 and temperature 14 and so on...
for index, temperature in enumerate(temperatures): 
    print("Temperature at "+str(hour_of_day[index]) + ":00: "+str(temperature) + "°C")
```

#### 4.2.2 The while-loop
A while loop is not used as often as a for-loop. In the definition you define a condition
and "while" that condition holds, the loop is executed. 

Look at this example:
```python
a = 1
while a <= 10:
    print(a)
    a = a +1
```
Can you guess what will be display?
<details>
<summary>Solution</summary>
It will print the numbers 1 to 10, including 10
</details>

{% capture warning %}

<h3> Warning </h3>
<p >When you define a while-loop, always make sure that the condition will at some
point be fullfilled. Otherwise it can easily happen that youre while-loop just keeps running
endlessly!</p>

{::options parse_block_html="true" /}

```python
a = 1
# This loop will run forever, because a will never be > 10!
while a <= 10:
    print(a)
```

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--warning">
  {{ warning | markdownify }}
</div>


<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}

<h3> Exercise </h3>
<p >Now you already know quite some tools for writing a Python script! Use your knowledge to complete the code below.
The goal is to print the sentence "{month} was a hot month" whenever the mean monthly temperature is above two times the mean
and "{month} was a dry month" whenever the precipitation was less than half of the mean.  </p>
<p >One tip: For the printing you can use formatted strings. They make inserting variables in a string much easier! 
just put an "f" in front of the string and insert the variable in curly braces {}.  
For example print(f"Hello {name}" would print "Hello Peter" if the variable name=Peter is defined.</p>
<p >Here is your starter code:</p>


{::options parse_block_html="true" /}

```python
months = ["January", "February", "March", "April", "May", "June", "Juli","August", "September", "October", "November", "December"]
monthly_temperature = [4, 4, 7, 11, 15, 17, 28, 24, 27, 12, 7, 4]
monthly_precipitation = [15, 40, 60, 75, 65, 32, 10, 80, 60, 70, 57, 100]

mean_temperature = 
mean_precipitation = 

for ... in enumerate(...):
    if ...:
        ...
    if ...:
        ...


```

<details><summary markdown="span">Solution!</summary>

```python
months = ["January", "February", "March", "April", "May", "June", "Juli","August", "September", "October", "November", "December"]
monthly_temperature = [4, 4, 7, 11, 15, 17, 28, 24, 27, 12, 7, 4]
monthly_precipitation = [15, 40, 60, 75, 65, 32, 10, 80, 60, 70, 57, 100]

mean_temperature = sum(monthly_temperature)/len(monthly_temperature)
mean_precipitation = sum(monthly_precipitation)/len(monthly_precipitation)

for index, month in enumerate(months):
    if monthly_temperature[index] > 2*mean_temperature:
        print(f"{month} was a hot month")
    if monthly_precipitation[index] < mean_precipitation/2:
        print(f"{month} was a dry month")
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>

## 5. Functions and Classes
**Congratulations!**  
You made it this far down, that means you have accquired knowledge of the basic building blocks of Python.
  
![Congrats Meme](assets/images/python/1/grats.gif)  

You are now ready to go into two concepts that go beyond basic scripting (meaning, just putting all your code line by line into one file),
and learn about the fundamental blocks that help strcuturing your program: Functions and Classes!

### 5.1 Functions
Functions are constructs of own, separate blocks of code in your program which take care of certain tasks. They are super useful, because often 
 you want to do the same operation many times in your code but don't want to write the same code every time again. Just write your own function and 
 call it whenever you need its expertise! Lets just look at a simple example:
 ```python
 def calculate_mean(list_of_values):
    n_samples = len(list_of_values)
    sum_of_values = sum(list_of_values)
    mean = sum_of_values/n_samples
    return mean
 ```
 Pretty intuitive, right? 
 
 A function is always defined by starting with the keyword "def", then we give it a name,
 calculate_mean in this case. Afterwards in the braces are the "arguments" that the function takes. Arguments are
 pieces of information from the outside code, which the function requires to work. Here it is the list_of_values
 the function shall calculate the mean value of. After the ":" we follow with the indented codeblock that belongs
 to the function. Here we do all the operations the function should do. Finally, we use the "return" keyword which ends
 the function and defines, which piece of information should be returned to the outside code. 
 
 > **Important** The variables which are defined inside a function are restricted to that function!
 > The outside code won't know of the variables n_samples or "mean" which are defined in the function.
 
 Calling the function would
 for example look like this:
 ```python
monthly_temperature = [4, 4, 7, 11, 15, 17, 28, 24, 27, 12, 7, 4]
mean_monthly_temperature = calculate_mean(monthly_temperature)
 ```
 You do not **have to** return a value. You could also for example print something in the function and then return, without
 providing a value to return.  
   
In older versions of Python this was all there was to writing a function. However, nowadays you can add some additonal
information to make it even easier for the next person or your future self to understand it. With some extra bits 
you can add the infos, what type of data you expect as an input to the function and what type of data it will output.
This is generally a good thing to do and now considered best practice when writing functions. For the above code it would
look like this:
 ```python
def calculate_mean(list_of_values:list[float]) -> int:
    n_samples = len(list_of_values)
    sum_of_values = sum(list_of_values)
    mean = sum_of_values/n_samples
    return mean
 ```
 In the first line, after the list_of_values we write ":list[float]" to specify that we 
 expect a list of float (floats actually imply integers, so we can use that to also accept integers).
 After the closing bracket we write  
 "-> int" which states that this function will return an integer value.  


<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}

<h3> Exercise 5.1.1 </h3>
<p >As a first exercise, try to figure out what the output of the below
function will be without executing it!</p>

{::options parse_block_html="true" /}

```python 
def square_value(value:int) -> int:
    return value * value

def divide_value_by(numerator:int, denominator:int) -> int:
    return numerator / denominator

a = square_value(2)
b = square_value(a)
c = divide_value_by(b,a)
d = square_value(divide_value_by(c,2))
print(d)
```


<details><summary markdown="span">Solution!</summary>
<p >The result is 4!</p>

```python
def square_value(value:int) -> int:
    return value * value

def divide_value_by(numerator:int, denominator:int) -> int:
    return numerator / denominator

a = square_value(2)                     # 2*2 = 4
b = square_value(a)                     # 4*4 = 16
c = divide_value_by(b,a)                # 16/4 = 4
d = square_value(divide_value_by(c,2))  # square_value gets the output of divide_value(c,2) as argument.
print(d)                                # so 4/2 is 2, thn 2*2 is 4

```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}

<h3> Exercise 5.1.2 </h3>
<p >Lets go for a bit more challenging of an exericse (I am sure you are ready for it!)
There is a built-in function that allows the user to give an input through the command-line to the program.
It is simply called "input()". E.g. "testword = input()" would stop the program and wait for the user to input
something in the console and then press enter. </p>
<p  >Imagine you want a program in which you set a new password.
Write a function that checks whether the new password is longer than 9 symbols and that returns the corresponding
boolean. the function should also print that the password is too short if it is too short and that it is ok when it is ok. 
Use the returned boolean to keep asking for new input from the user <u>while</u> the word is less than 9 characters long</p>
<p  >Here is some starter code:</p>

{::options parse_block_html="true" /}

```python 
def is_password_too_short(word:str, min_length:int)->bool:
    is_password_too_short = ...
    if ...:
        ....
    else:
        ....
    return ...

password_is_bad = True
while ...:
    print("Please enter your password:")
    password = input()
    password_is_bad = ...
```


<details><summary markdown="span">Solution!</summary>

```python
def is_password_too_short(word:str, min_length:int)->bool:
    is_password_too_short = len(word) < min_length
    if is_password_too_short:
        print(f"Password has to be at least {min_length} characters long!")
    else:
        print("New password set!")
    return is_password_too_short

password_is_bad = True
while password_is_bad:
    print("Please enter your new password:")
    password = input()
    password_is_bad = is_password_too_short(password, 8)
 ```

</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>

### 5.2 Classes
Classes are the final fundamental building block of Python we will look at here.
A class basically represents a blueprint of an object that has certain properties. As an example, 
when I am working with data on ecosystems it could be convenient to have an ecosystem class
that includes information about the ecosystem type, the location as latitude and longitude, and
some meteorological data. Lets look at an example:
```python
class Ecosystem():
    def __init__(self, id, IGBP_ecosystem_class, lat, lon, mean_annual_Tair, mean_annual_precip):
        self.id = id
        self.IGBP_ecosystem_class = IGBP_ecosystem_class
        self.lat = lat
        self.lon = lon
        self.mean_annual_Tair = mean_annual_Tair
        self.mean_annual_precip = mean_annual_precip
```

The definition of a class always begins with the class-keyword followed by the name of the class.
It always has a first function called __init__() which is also called "constructor". This method is used to 
create new instances of the class and assigns values to the class. The "self" keyword is in this context 
always used within a class to reference the class itself. Note, that "self" also has to be in the list of arguments
for the function, but is does not get passed when you call the function.  
Many new words but stay with me, it is pretty simple when we look at an example, how we create a new instance:
```python
# First we define the class
class Ecosystem():
    # see how the first argument here is "self"
    def __init__(self, id, IGBP_ecosystem_class, lat, lon, mean_annual_Tair, mean_annual_precip):
        self.id = id
        self.IGBP_ecosystem_class = IGBP_ecosystem_class
        self.lat = lat
        self.lon = lon
        self.mean_annual_Tair = mean_annual_Tair
        self.mean_annual_precip = mean_annual_precip

# now we use that class to create a new ecosystem-object, 
# see how we have to provide every value defined in the constructor except for "self":
amtsvenn = Ecosystem(id="amtsvenn", IGBP_ecosystem_class = "open shrublands", lat = 52.176, lon = 6.955, mean_annual_Tair = 10.5, mean_annual_precip = 870)

# Now you have stored all the info about amtsvenn in the "amtsvenn"
# object and can access them whenever you want:
print(amtsvenn.id)
print(amtsvenn.IGBP_ecosystem_class)
print(amtsvenn.lat)
print(amtsvenn.lon)
```
Classes can not only comprise of the information associated with them but
can also have methods associated specifically with them. For example we can create
a function that prints all the information enclosed in the object. 
```python

class Ecosystem():
    def __init__(self, id, IGBP_ecosystem_class, lat, lon, mean_annual_Tair, mean_annual_precip):
        self.id = id
        self.IGBP_ecosystem_class = IGBP_ecosystem_class
        self.lat = lat
        self.lon = lon
        self.mean_annual_Tair = mean_annual_Tair
        self.mean_annual_precip = mean_annual_precip

    def print_ecosystem_information(self):
        print("=====================")
        print("Ecosystem information")
        print(f"ID: {self.id}")
        print(f"IGBP ecosystem class: {self.IGBP_ecosystem_class}")
        print(f"Location (lat/lon): {self.lat}°/{self.lon}°")
        print(f"Mean annual air temperature: {self.mean_annual_Tair} °C")
        print(f"Mean annual precipitation: {self.mean_annual_precip} mm")

amtsvenn = Ecosystem(id="amtsvenn", IGBP_ecosystem_class = "open shrublands", lat = 52.176, lon = 6.955, mean_annual_Tair = 10.5, mean_annual_precip = 870)

# After creating the object we can use the classes functions like this:
amtsvenn.print_ecosystem_information()
```

<div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
{% capture exercise %}

<h3> Exercise 5.2.1 </h3>
<p >Lets do one exercise that can further show, why classes are great for creating reusable code. 
Try to write a function called "Statistics". This class will be a "behavioural" class, meaning it does not need to hold
own data but rather holds some methods, that belong to the same topic. In that class, define functions that calculate the 
mean, the variance and the standard deviation of a given list. Then use that class to calculate these metrics of an arbitrary list. </p>
<p > Hint: For the standard deviation you need to take the square root. You can do that with pythons built-in math module.
You can use it like this: </p>
```python
import math
math.sqrt(24)
```
<p >Try to work out the solution yourself first! There is some starter code below, in case you get stuck though.</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Starter code</summary>
```python
class Statistics():
    
    def calculate_mean(self, ...):
        ...
    
    def calculate_variance(self, ...):
        mean = ...
        squares = []
        for value in values:
            squares.append(...)
        variance = ...
        return ...
    
    def calculate_stdev(self, ...):
        variance = ...
        stdev = ....
        return ...
```
</details>

<br>

<details><summary markdown="span">Solution!</summary>

```python
import math

class Statistics():
    
    def calculate_mean(self, values:list[float]):
        return sum(values)/len(values)
    
    def calculate_variance(self, values:list[float]):
        mean = self.calculate_mean(values)
        squares = []
        for value in values:
            squares.append((value-mean)**2)
        variance = sum(squares) / (len(values)-1)
        
        return variance
    
    def calculate_stdev(self, values:list[float]):
        variance = self.calculate_variance(values)
        stdev = math.sqrt(variance)
        return stdev
    
stat = Statistics()
example_list = [1,2,3,4,5,5,6,7,123,1,1,4]
mean = stat.calculate_mean(example_list)
stdev = stat.calculate_stdev(example_list)
variance = stat.calculate_variance(example_list)
```

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
</div>

We will not go deeper into classes here, but it is very important to understand the concept. Most 
Python packages are written in object-oriented style, which (in very simple terms) means that the
methods are enclosed in classes. So knowing the basics makes it much easier to understand the following bits.
