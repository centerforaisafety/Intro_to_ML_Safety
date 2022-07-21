Adversarial Examples
====================

**Contributor(s): Oliver Zhang**

Distributional shifts or black swan events are not the only dangers when
deploying machine learning systems. In the real world, there exist
people with a wide range of intents, some of whom will act as
adversaries, purposefully trying to make systems misbehave. There are
many possible reasons for this—perhaps the adversary is designing
malware to avoid machine learning malware detection algorithms. Perhaps
the adversary wants to design sound files which can give unwanted
instructions to voice-recognition systems (e.g., Google Home, Amazon
Echo). As machine learning systems become more and more integrated into
daily life, the incentives to attack these systems only grow larger over
time.

We define *adversarial attack* against a machine learning system as a
technique which causes a machine learning system to misbehave. Often
adversarial attacks revolve around the generation of *adversarial
examples*, inputs designed to confuse or deceive the machine learning
system. The literature around adversarial examples seeks to answer three
questions regarding these adversarial attacks and examples.

-   First, can we discover potential adversarial attacks and reveal
    vulnerabilities in our machine learning systems?

-   Second, can we train systems to be robust to adversarial attacks?

-   Third, can our knowledge of these strange failure cases inform our
    understanding of machine learning systems?

Our approach to introducing this field is as follows. In subsection
1.1 we introduce the basic *L*<sub>*p*</sub> adversarial
examples paradigm. In subsections 1.2 and 1.3, we introduce two classic adversarial attacks, namely
the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent
(PGD). In subsection 1.4, we introduce adversarial
training, a basic defense against adversarial attacks. In subsection 1.5, we discuss two surprising properties of adversarial
attacks, specifically their transferability 1.5.1 and
their existence in the real world in 1.5.2. Finally, in
subsection 1.6, we go beyond the *L*<sub>*p*</sub>
paradigm and discuss other adversarial attacks which don’t fit in this
paradigm.

*L*<sub>*p*</sub> Adversarial Robustness
----------------------------------------

Adversarial examples were first discovered by scientists at Google, who
noticed that you could add imperceptible noise to an image classifier
and make the classifier change its behavior (see figure
<a href="#fig:openai_panda" data-reference-type="ref" data-reference="fig:openai_panda">1</a>)
(Szegedy et al. 2014). To humans, the image looked exactly the same, but
to the model, the image had changed tremendously. As further scientists
explored this phenomena, their work grew into one of the first
adversarial attacks paradigms, the “*L*<sub>*p*</sub> adversarial
robustness paradigm.” Within this paradigm, we generate adversarial
examples by tweaking existing training examples in a targeted fashion.
Moreover, this paradigm requires the adversarial example to be ‘close’
to its corresponding training example. We defined this ‘closeness’
through some norm from the *L*<sub>*p*</sub> family of norms (aka the
*p*-norm). Two common *p*-norms are the 2-norm, which is just the
euclidean distance and the ∞-norm, which is simply the maximum over the
dimensions.

<p align="center">
<img src="images/openai_panda.png" id="fig:openai_panda" style="width:10cm" alt="By adding imperceptible noise to an image, we can fool the model into classifying the image improperly." /><figcaption aria-hidden="true">Figure 1: By adding imperceptible noise to an image, we can fool the model into classifying the image improperly.</figcaption>
</p>

We restate the paradigm using mathematical notation. Let $\mathcal{D}$ be a dataset
and *x* ∈ $\mathcal{D}$ be an example in our training dataset. Let *f* : 𝒳 → 𝒴 be a
machine learning model (usually a classifier) which we are trying to
disrupt. An adversarial attack consists of defining a function
*g*<sub>*a**d**v*</sub> : 𝒳 → 𝒳 which maps training examples to
adversarial examples under the constraint that
∥*g*<sub>*a**d**v*</sub>(*x*) − *x*∥<sub>*p*</sub> &lt; *ε*. Here *ε*
represents a ‘budget’ on how far the adversarial example can deviate
from the initial training example and is defined before we choose an
attack. The stronger the attack, the more drastically
*g*<sub>*a**d**v*</sub>(*x*) changes the behavior of our model *f*,
while being close to *x* (i.e., with *ε* being relatively small). With
this framework in place, we can start studying some simple adversarial
attacks.

Fast Gradient Sign Method
-------------------------

The Fast Gradient Sign Method (FGSM) is one of the earliest techniques
in generating adversarial examples (Goodfellow, Shlens, and Szegedy
2015). It was developed in 2015 and gained widespread attention within
the community for being a simple and quick way to generate adversarial
examples. It operates on the *L*<sub>∞</sub> norm, meaning that each
pixel in the images it generates differs at most from the original image
by some *ε*.

Let *l* be some loss function (e.g., cross entropy loss) used to train
*f*. Moreover, let *ŷ* = *f*(*x*) be the predictions of our model on
input *x*. Then, we can generate adversarial examples as follows:

$$*x_{adv}* = *x* + *ε* \* sign(∇_{*x*}(*l*(*ŷ*, *y*))).$$

The gradient ∇<sub>*x*</sub>(*l*(*ŷ*, *y*)) represents how to modify *x*
to maximize the loss *l*(*ŷ*, *y*). We then take the sign of the
gradient before adding it to the original example *x*. This does two
things. First, it helps bound the gradient between -1 and 1. This
bounding of the gradient makes *x*<sub>*a**d**v*</sub> naturally satisfy
the ∞-norm constraint:

$$*x_{adv}* − *x* = *ε* \* sign(∇_{*x*}(*l*(*f*(*x*), *y*)).$$

Second, taking the sign of the gradient actually works better than
naively using the gradient, so long as *f* can be approximated well as a
linear function around the input (Goodfellow, Shlens, and Szegedy 2015).
Intuitively, taking the sign of the gradient maximally uses up the
∞-norm constraint.

In practice, FGSM is a relatively weak method for generating adversarial
examples and can be defended against easily. That being said, FGSM often
can still deceive models which have not specifically implemented any
adversarial defenses. Moreover, FGSM’s simplicity makes it a useful
baseline attack to expand upon or compare against.

Projected Gradient Descent
--------------------------

After FGSM, we jump to Projected Gradient Descent (PGD), which Madry et
al. adapted for adversarial attacks (Madry et al. 2019). Whereas FGSM
was analogous to taking one step of gradient ascent, PGD is analogous to
taking multiple steps of gradient ascent. Let *x*<sub>*i*</sub>
represent the adversarial example after *i* steps of PGD. Let *α* be a
‘step size,’ proj a function which projects an *x*′ to be within *ε*
*p*-norm of the original *x* and *n* be the total number of steps. Then
we repeat the following procedure for *i* = 1 to *i* = *n*:

$$\\begin{aligned}
    x\_{i+1}' &= x\_i + \\alpha \* \\nabla\_x(l(f(x), y)) \\\\
    x\_{i+1} &= \\text{proj}(x\_{i+1}')\\end{aligned}$$

In practice, PGD generates very strong adversarial examples, as *n*
increases. There are many adversarial defenses which are broken by
simply increasing the number of steps, *n*. On the flipside, the
strength of PGD’s adversarial examples gives us a way to train models to
be robust.

Adversarial Training
--------------------

Now that we’ve learned about attacks like FGSM or PGD, one natural
question might be to ask: “how might we train models to be *robust* to
adversarial attacks?” One common approach is adversarial training.
During adversarial training, we expose our model to adversarial examples
and penalize our model if the model is decieved. In particular, an
adversarial training loss might be as follows:

$$\text{Loss}(f, \mathcal{D}) = \underset{x,y~ \sim \mathcal{D}}{\mathbb{E}}\left[\text{CrossEntropy}(f(x), y) + \lambda \cdot \text{CrossEntropy}(f(g_{adv}(x)), y)\right].$$

where *λ* is some hyperparameter determining how much we emphasize the
adversarial training. This often reduces accuracy, but increases
robustness to the specific adversarial attack which your model is
trained on. However, many have shown that robustness towards one type of
adversarial attack does not provide robustness to other adversarial
attacks. For instance, a model trained on FGSM will not perform well on
PGD attacks with *n* = 10. PGD was significant because PGD showed that
models trained on actually were robust to a whole class of adversarial
examples.

Properties of Adversarial Examples
----------------------------------

Adversarial examples also exhibit some special properties. Specifically,
adversarial examples are moderately transferable to different models
trained on the same dataset (or even on different datasets).
Additionally, adversarial examples can be printed out and fool
real-world systems through a camera lense. We describe these findings
and their implications in this section.

### Transferability of Adversarial Examples

First, (Szegedy et al. 2014; Goodfellow, Shlens, and Szegedy 2015)
demonstrated that adversarial examples from one model are actually
somewhat transferable to other models, including models which were
trained on a disjoint training set. The precise accuracy varies
depending on attack method, model architecture, and datasets.

This itself has lead to a new type of adversarial attack, which can be
performed without access to the actual model. It involves training a
surrogate model, attacking that surrogate model, and hoping the attack
transfers to the unseen model (Papernot, McDaniel, and Goodfellow 2016;
Papernot et al. 2017).

### Real World Adversarial Examples

Second, Kurakin et al. demonstrated that adversarial examples can be
printed out and fool real-world systems through a camera lens (Kurakin,
Goodfellow, and Bengio 2017). This serves to demonstrate the adversarial
examples are robust to noise and perturbation themselves. Soon after,
Brown et al. developed ‘adversarial patches,’ paper printouts which can
be added to a video feed to attack any image classifiers monitoring the
video feed (Brown et al. 2018). Others have designed ‘toxic signs’ which
deceive autonomous driving systems (Sitawarin et al. 2018), stickers
which fool facial recognition systems (Komkov and Petiushko 2021), and
sweaters which fool object detectors (Wu et al. 2020).

Unforseen Attacks
-----------------

\[Unfinished\]

References
==========

Brown, Tom B., Dandelion Mané, Aurko Roy, Martín Abadi, and Justin
Gilmer. 2018. “Adversarial Patch.” <http://arxiv.org/abs/1712.09665>.

Croce, Francesco, Maksym Andriushchenko, Vikash Sehwag, Edoardo
Debenedetti, Nicolas Flammarion, Mung Chiang, Prateek Mittal, and
Matthias Hein. 2021. “RobustBench: A Standardized Adversarial Robustness
Benchmark.” <http://arxiv.org/abs/2010.09670>.

Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. 2015.
“Explaining and Harnessing Adversarial Examples.”
<http://arxiv.org/abs/1412.6572>.

Komkov, Stepan, and Aleksandr Petiushko. 2021. “AdvHat: Real-World
Adversarial Attack on ArcFace Face ID System.” *2020 25th International
Conference on Pattern Recognition (ICPR)*, January.
<https://doi.org/10.1109/icpr48806.2021.9412236>.

Kurakin, Alexey, Ian Goodfellow, and Samy Bengio. 2017. “Adversarial
Examples in the Physical World.” <http://arxiv.org/abs/1607.02533>.

Madry, Aleksander, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras,
and Adrian Vladu. 2019. “Towards Deep Learning Models Resistant to
Adversarial Attacks.” <http://arxiv.org/abs/1706.06083>.

Papernot, Nicolas, Patrick McDaniel, and Ian Goodfellow. 2016.
“Transferability in Machine Learning: From Phenomena to Black-Box
Attacks Using Adversarial Samples.” <http://arxiv.org/abs/1605.07277>.

Papernot, Nicolas, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z.
Berkay Celik, and Ananthram Swami. 2017. “Practical Black-Box Attacks
Against Machine Learning.” <http://arxiv.org/abs/1602.02697>.

Sitawarin, Chawin, Arjun Nitin Bhagoji, Arsalan Mosenia, Mung Chiang,
and Prateek Mittal. 2018. “DARTS: Deceiving Autonomous Cars with Toxic
Signs.” <http://arxiv.org/abs/1802.06430>.

Szegedy, Christian, Wojciech Zaremba, Ilya Sutskever, Joan Bruna,
Dumitru Erhan, Ian Goodfellow, and Rob Fergus. 2014. “Intriguing
Properties of Neural Networks.” <http://arxiv.org/abs/1312.6199>.

Wu, Zuxuan, Ser-Nam Lim, Larry Davis, and Tom Goldstein. 2020. “Making
an Invisibility Cloak: Real World Adversarial Attacks on Object
Detectors.” <http://arxiv.org/abs/1910.14667>.
