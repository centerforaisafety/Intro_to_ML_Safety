![](images/DALL·E%202023-05-25%2015.50.01%20-%20synthwave%20mural%20of%20puzzle,%20scientific,%204k,%20futuristic,%20neon.png)

# Risk Decomposition
**Contributor(s): Cody Rushing**

*Note: these notes largely replicate what associated video says, but have a few additional comments and links to examples (Cody encourages people to add more information as they see fit). They are written and organized to (hopefully) provide a different perspective on the content, and to clear up any confusions the video may have posed.*

To effectively approach making systems safer, we need some sort of structure to guide our thinking. Making Machine Learning systems safer isn't trivial; It's not exactly like fixing a car, nor is it exactly like debugging code. 

Here are some key definitions to motivate our thinking:
- **Failure Mode**: One possible way in which a system might fail (example: a system crashing would fail to complete it's intended job)
- **Hazard**: A source of danger with the potential to harm (example: an AI with the goal to turn the world into Paperclips)
- **Vulnerability**: A factor or process that increases susceptibility to the damaging effects of hazards (example: if an AI system is *brittle*, meaning that it can only recognize previously seen patterns[^1], it is *vulnerable* as it has high susceptibility to exploits.)
- **Threat:** A hazard with the intent to exploit a vulnerability (example: an AI intending to maximize Paperclips by hacking into various servers with known vulnerabilities)
- **Exposure**: the extent to which elements (e.g. people, property, systems, etc.) are subjected to hazards (example: while someone texting while driving is dangerous, the exposure is constrained to the people/objects currently around the driver)
- **Ability to Cope**: the ability to efficiently recover from the effects of hazards

## Decompositions of Risk
We can use hazards to define a simple definition of risk. Given a set of hazardous events $H$ we are concerned about, the risk of them can be defined as:
$$Risk = \sum _{h \in H}P(h)Impact(h)$$
This is overly-simple, though, and doesn't really think about hazards in context. For a more defined and notional understanding of risk, we can decompose a risk notionally as $Risk \approx Vulnerability \times Exposure \times Hazard$. Specifically, given a set of Hazardous events $H$ and a nonlinear interaction $\times$, this notional decomposition frames risk to be:$$Risk = \sum _{h \in H} P(h) \times Severity(h) \times P(Contact|h) \times Vulnerability(h)$$
We introduce $Vulnerability(h)$ and $P(Contact|h)$ as it's not enough to merely think about hazards by themselves: we need to place them in context of the our total exposure to them and how vulnerable our systems are to them. Notice that we've formalized that $Exposure(h)=P(Contact|h)$. This definition is more holistic, considering:
- The probability of impact: $P(h)$ and $P(Contact|h)$
- The impact itself: $Severity(h)$ and $Vulnerability(h)$

### Examples

Here are three examples of risks, broken down into their notional components:

|                                  | Vulnerability                              | Hazard Exposure                              | Hazard                      |
|----------------------------------|--------------------------------------------|----------------------------------------------|-----------------------------|
| Falling on a wet floor           | Bodily Brittleness                         | Floor Utilization                            | Floor Slipperiness          |
| Damage from Flooding             | Low Elevation, Unsuspended Buildings, etc. | Number of People and Worth of Assets in Area | Rainfall Volume             |
| Flu-Related Health Complications | Old Age, Poor Health, etc.                 | Contact with Flu Carriers                    | Flu Prevalence and Severity |


Now that we have decomposed the risk into their components, we can systematically consider how we can reduce the risk. For example, for the case of falling on a wet floor, we could:
- Address the **hazard** itself by introducing a fan and making the floor less slippery
- Reduce the **exposure** to the hazard by putting up a "Caution, Slippery When Wet" Sign
- Or, we could reduce our **vulnerability** to the risk by undergoing strength training and making our body less brittle

Or, in the case of flu-related health complications, we could:
- Address the **hazard** itself by cleaning surfaces to reduce the flu prevalence
- Reduce the **exposure** by reducing contact with Flu carriers
- Decrease **vulnerability** to the flu by encouraging vaccinations

## How should we reduce risks for Machine Learning Systems?

Again, we can break down the risk posed by Machine Learning systems into the same components.

For instance, most of the 'Hazard' portion of the risk posed by ML systems stems from the **Alignment problem**; Alignment Researchers consider how to reduce the probability and severity of inherent model hazards. One research direction in the Alignment landscape is focused on how to constrain Agents to only pursue good goals. This is often broken up into two distinct subproblems, the [Inner and Outer Alignment problem](https://www.lesswrong.com/posts/poyshiMEhJsAuifKt/outer-vs-inner-misalignment-three-framings-1) (one which considers how to establish a good goal, the other which considers how to ensure the system robustly adopts this goal).

The Vulnerability portion of the risk can be characterized by the **Robustness** research field. Robustness considers how we can withstand hazards (and thus, how we can reduce our vulnerability to them).

Finally, the Hazard Exposure portion of the risk is characterized by the **Monitoring** research field, which is concerned with identifying hazards. When we can identify a potential hazard, the risk gets lower as we can take steps to limit our exposure to them. A relevant example of this is the work [ARC Evals ](https://evals.alignment.org/) focuses on, and their role in evaluating GPT-4 to attempt to detect hazards[^2].

The broader field of Systemic Safety is concerned with how to simultaneously reduce multiple components of the risk factor equation simultaneously, and is focused on reducing systemic risks. These concern areas such as ML for Improved Epistemics, ML for Improved Cyberdefense, and Cooperative AI.

## Better Understanding Existential Risks

There is one final factor we should factor into our risk equation: our ability to cope. This introduces the **Extended Disaster Risk Equation**:
$$Risk \approx \frac{Vulnerability \times Exposure \times Hazard}{Ability \ To \ Cope}$$

Notice here that if our ability to cope approaches zero, the total risk scales to infinity. This is an important consideration for advanced AI: if we have an AI that is both more powerful than all other models combined and unaligned with our values, our ability to gain back control would be low, and so our ability to cope would be near zero. This would, consequently, be a *high risk scenario.*

This serves as a useful intuition pump for understanding why we wish to avoid x-risks: they remove the ability to cope.

## Reducing Risk vs Estimating Risk

There is an important distinction between estimating and reducing risk that needs to be made: In practice, the actions to estimate and reduce risk are different, and as such, there is a critical tradeoff that occurs between the two. It may be alluring to spend resources to gain a precise estimate of risk, but in situations where we already know it is high-stakes, we may often want to just focus on reducing the risk. There are a few reasons why this may occur:
- The situation may have high inherent uncertainty, and so making the estimate more precise may not be possible
- The precision increases are not action-relevant; a hazard being 85% vs 86% likely may not cause one to heavily update on the best course of action

In these situations, it is far more valuable to instead identify ways to reduce vulnerabilities, exposure, and hazards (and *then* futher prioritize, respond, and assess the hazard). We don't want to burn time getting a more precise estimate of the risk if it causes us to be unable to address it.

[^1]: See [this paper](https://arxiv.org/abs/1811.11553) for an example of brittleness in AI image classifiers, who fail to classify rotated images
[^2]: See part D, section 1.1: "Overview of findings and mitigations" of the [Technical GPT-4 report](https://arxiv.org/pdf/2303.08774.pdf) to see a broad overview of this