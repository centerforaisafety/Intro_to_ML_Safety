# Black Swans

## Overview
Risk estimation in the real world is messy, and black swans have a large impact on it.
> “Things that have never happened before happen all the time.” Scott D. Sagan

This lecture is abstract and the content is hard to internalize as it is very deep.
It provides a vocabulary for talking about extreme events, which will help provide a structure for thoughts and analysis.
## Roadmap
- Black Swans
- Long Tailed Distributions
- Thin Tailed Distributions
- Mediocristan and Extremistan
- Unknown Unknowns
- Long Term Safety
## Black Swans
Black Swans are named as such due to the discovery of black swans in Australia, at a time when Western belief was that there were only white swans in existence[^1]. 

![Image of a black swan](https://upload.wikimedia.org/wikipedia/commons/0/0a/Black_Swan_2_-_Pitt_Town_Lagoon.jpg)

In short, Black Swans are things that happen that we don't see coming. They are outlier events that, although unexpected, can have extreme impacts. Machine learning and artificial intelligence rely on the certainty of knowledge, and Black Swans generate uncertainty.
```
Example: Black Swan Events in Autonomous Vehicles
- Poor weather and lighting conditions
- Reflections and/or obstructions
- People in costumes
- Animals in road
```
Accounting for this uncertainty created by Black Swan events is therefore just as important as accounting for everything that is certain. One article handles Black Swan events by introducing a diversity term in the loss function that is used to train a neural network model. Adding this term improved the accuracy of the algorithm's response to Black Swan events. Read the article here: [Handling Black Swan Events in Deep Learning with Diversely Extrapolated Neural Networks](https://www.ijcai.org/proceedings/2020/0296.pdf)[^2]
## Long Tail Distributions
Black Swans are usually considered long tail events.
- A long tail distribution is a case where the tail (the region far from the center of the distribution) tapers off gradually from the head (center) of the distribution[^3].
  - Examples: power law distribution, pareto distribution [^4]

![Image of a general long tail distribution](https://miro.medium.com/max/1400/1*4Y4iR2HQL5lPpHN46WqsSQ.png)

Long tails are normally characterized as max-sum equivalent, which means that larger events dominate over the sum of the other events. They render normal statistical analysis useless, as tools such as mean, standard deviation, etc. are not feasible for use with long-tailed data.

![Screenshot (10)](https://user-images.githubusercontent.com/106725257/186303322-9f0204f0-a5d0-42c7-b7ef-5009b4403ad5.png)
```
Real world examples: word frequency (few words are used far more often than the majority of other words), website hits (sites like
google are used much more than others), net worth (the few richest people are worth more than the majority of the population combined)
```
Nonlinear interactions generate long tailed distributions (while linear interactions create a thin tailed gaussian).
This can happen when parts of the interaction are dependent on one another (meaning if the result becomes zero if just one of the parts becomes zero).
Read more on long tail distributions and examples of their effects on AI models here: [AI Economics, Synthetic Data & the Long Tail. | by Sushrut Mair | Dev Genius](https://blog.devgenius.io/ai-economics-synthetic-data-the-long-tail-ed23f460a42a) [^5]

## Thin Tail Distributions
Exponential and gaussian distributions have thin tails, as the probability density drops off exponentially fast (e<sup>-x</sup> and e<sup>-x<sup>2</sup></sup>, respectively). Because the probability density drops off so quickly, it becomes far less likely for these distributions to have an extreme event than for a long tail to have an extreme event (See graphs below).
![Screenshot (12)](https://user-images.githubusercontent.com/106725257/186304091-7da81dd0-e226-4336-9761-698636bca8fb.png)

## Mediocristan and Extremistan
These are concepts thought of by Nassim Nicholas Taleb, a mathematical statistican and risk analyst who disagreed with the idea of using a bell curve analysis for risk mitigation and analysis because it ignored black swan events by assuming most things lie in the average of the bell curve [(read more here)](https://people.wou.edu/~shawd/mediocristan--extremistan.html) [^6].
### Mediocristan
The main impact of Mediocristan is that the impact of an outlier is insignificant when compared to the total (tyranny of the collective). For example, if you imagine there are a bunch of people grouped together, and you choose the heaviest person and get their weight, it will be very small relative to the sum of everybody’s weight.

Because each individual event is likely to be average, you can use normal statistical analysis tools. Mediocristan is used synonymously with thin tailed to describe events.
### Extremistan
The impact of an outlier in Extremistan is very large when compared to the total (tyranny of the accidental). For example, if you were to look at all the people on the planet and add up their net worth, Bill Gates would still make up a large percentage of the total wealth because he has much more money than the average person.

Taleb claims that almost all social matters are from Extremistan, while physical matters such as weight, height, and calorie consumption are from Mediocristan.

Recognizing whether it is mediocristan or extremistan is an important part of risk analysis because it informs what conditions are present (ie. can we use normal statistical analysis tools and how much do we need to take extreme cases into account).
## Unknown Unknowns
### Known Knowns
- Things we know and understand (ie. facts).
### Known Unknowns
- These are things that we know exist, but we classify as risks because of our lack of understanding
- Can be better understood with closed-ended questions
### Unknown Knowns
- These are things that we are unaware of, but wouldn’t be caught off guard if they became known (ie. unaccounted facts).
### Unknown Unknowns
- Things we are unaware of and don’t understand (unknown risks)
- Can be encountered through open-ended exploration of all possible situations (explanations of finding unknown unknowns can be read more about [here](https://hbr.org/2017/10/simple-ways-to-spot-unknown-unknowns)) [^7]
```
Black Swans can be known unknowns, but they are usually unknown unknowns
```
## Long-Term Safety
AI needs to be able to account for Black Swans to minimize their impact, as AI’s impact will most likely be long tailed. Having a dataset of knowledge and one of unknowns (an exploration of possibilities), and having these datasets work together, is one approach to detect a Black Swan and prepare against it [(read more here)](https://www.maintworld.com/Asset-Management/Black-Swans-in-Maintenance-and-Industrial-AI-Predicting-the-Unpredictable) [^8]. Models for machine learning and AI must be able to predict and relearn using data analysis to minimize risk and to mediate consequences [(read more here)](https://www.researchgate.net/profile/Alfonso-Gonzalez-Briones/publication/326309962_Machine_Learning_Predictive_Model_for_Industry_40_13th_International_Conference_KMO_2018_Zilina_Slovakia_August_6-10_2018_Proceedings/links/5b601342aca272a2d6768eab/Machine-Learning-Predictive-Model-for-Industry-40-13th-International-Conference-KMO-2018-Zilina-Slovakia-August-6-10-2018-Proceedings.pdf) [^9].

![Screenshot (16)](https://user-images.githubusercontent.com/106725257/186304577-aca37265-bdea-49b5-bc2c-cb7f40fef451.png)

## References
[^1]: http://blackswanevents.org/?page_id=26#:~:text=The%20term%20Black%20Swan%20originates,history%20and%20profoundly%20changed%20zoology.
[^2]: https://www.ijcai.org/proceedings/2020/0296.pdf
[^3]: https://www.statology.org/long-tail-distribution/
[^4]: http://math.bme.hu/~nandori/Virtual_lab/stat/special/Pareto.pdf
[^5]: https://blog.devgenius.io/ai-economics-synthetic-data-the-long-tail-ed23f460a42a
[^6]: https://people.wou.edu/~shawd/mediocristan--extremistan.html
[^7]: https://hbr.org/2017/10/simple-ways-to-spot-unknown-unknowns
[^8]: https://www.maintworld.com/Asset-Management/Black-Swans-in-Maintenance-and-Industrial-AI-Predicting-the-Unpredictable
[^9]: https://www.researchgate.net/profile/Alfonso-Gonzalez-Briones/publication/326309962_Machine_Learning_Predictive_Model_for_Industry_40_13th_International_Conference_KMO_2018_Zilina_Slovakia_August_6-10_2018_Proceedings/links/5b601342aca272a2d6768eab/Machine-Learning-Predictive-Model-for-Industry-40-13th-International-Conference-KMO-2018-Zilina-Slovakia-August-6-10-2018-Proceedings.pdf


