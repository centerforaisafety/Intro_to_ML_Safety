# Cooperative AI
**Contributor(s): Bilal Chughtai**

Cooperative AI is an emerging research area even within safety. As such, this lecture focuses on motivation, background concepts, and promising future directions, over empericial papers.

# Motivation

Problems of cooperation — in which agents have opportunities to improve their joint wellbeing, but are not easily able to do so — are ubiquitous and important. Day to day examples include driving and meeting schedhuling, but the principles scale all the way up to many of the *world's most pressing problems*. Examples include
* Climate change is a problem where everyone contributes, but rational entities may not be incentivised to take action. Coordination is therefore required so that everyone may equally share the burden of action - a **collective action problem**.
* War is a problem where actors need to coordinate to prevent risks of escalation, which results in a worse outcome for both parties. This is an example of a **prisoners dilemma**.
* Curtailing the spread of a deadly pandemic problem requiring the coordination of every individual in a society.

Equally, human civilization and the success of the human species depends on our ability to cooperate. Some of our *greatest successes* can be attributed to cooperation
* Trade allowed individuals to specialise into ....
  
As our societies, economies, and militaries become ever more powerful and connected, the need for cooperation becomes greater. Advanced AI will only exarcerbate this problem, increasing power and interconnectedness further. We've seen examples of this already - the 2010 flash crash was a trillion dollar market crash caused by errors with artificial intelligence powered automated trading systems (though the precise reasons are still not well understood).

Cooperative AI seeks to reverse this problem. It asks *how can we use advances in AI to help us solve cooperation problems?* It aims to improve the ability of AI systems to foster cooperation between
* humans
* machines
* organisations
It improves ML safety through reducing risks from cooperation failures. 



# Mathematical Preliminaries

In order to do this, we first require some mathematical framework to think about cooperation. **Game theory** is the study of mathematical models of strategic interactions among rational agents, and encompasses a large range of possible cooperation problems.

**Definition:** A *game* is a triplet $\mathcal{G} = \left(I, (S_i)_{i\in I}, (h_i)_{i\in I}\right)$ where

1. $I$ is the set of players $I=\{1,...,n\}$
2. For each $i\in I$, $S_i$ is player $i$'s set of strategies and $S=\prod_i S_i$
3. For each $i\in I$, $h_i: S\rightarrow \mathbb{R}$ is a function mapping overall strategies to payoffs. One can decompose this into a function $g: S\rightarrow A$ from strategies to outcomes in some outcome space $A$, and then utiltity functions $u_i: A \rightarrow \mathbb{R}$ mapping from outcomes to real numbers. Then $\forall s\in S$, $h_i(s) = u_i (g(s))$.

In words, a game defines a set of possible strategies and payoffs for each agent.

## Non Cooperative Games

**Example 1:** The Prisoner's Dilemma. Suppose two members of a criminal gang are arrested and imprisoned. Each prisoner is in solitary confinement with no means of speaking to or exchanging messages with the other. The police admit they don't have enough evidence to convict the pair on the principal charge. They plan to sentence both to a year in prison on a lesser charge. Simultaneously, the police offer each prisoner a Faustian bargain. The possible outcomes are
The possible outcomes are:
* If A and B each betray the other, each of them serves two years in prison
* If A betrays B but B remains silent, A will be set free and B will serve three years in prison
* If A remains silent but B betrays A, A will serve three years in prison and B will be set free
* If A and B both remain silent, both of them will serve one year in prison (on the lesser charge).

We can represent these outcomes in a **payoff matrix**

|    A\B    	| cooperate 	| betray 	|
|:---------:	|:---------:	|:------:	|
| cooperate 	|   -1, -1  	|  -3, 0 	|
|   betray  	|   0, -3   	| -2, -2 	|

TODO: INSERT PRISONERS DILEMMA PAYOFF

We see betraying a partner offers greater reward than cooperating with them, so all purely rational self-interested prisoners will betray the other, even though mutual cooperation would yield a greater reward. 

**Definition:** A *Nash equilibrium* is a collection $(s_i)_{i\in I}$ with $s_i \in S_i$ of strategy profiles, such that no agent can benefit by unilaterally deviating.

That is, if Alice and Bob choose strategy A and B respectively, (A, B) is a Nash equilibrium if Alice has no other strategy available that does better than A at maximizing her payoff in response to Bob choosing B, and Bob has no other strategy available that does better than B at maximizing his payoff in response to Alice choosing A. 

**Definition:** A *dominant strategy* is a strategy that is better than another strategy for one player, no matter how that player's opponents may play. It follows that if a strictly dominant strategy exists for one player in a game, that player will play that strategy in each of the game's Nash equilibria.

Given finite games, one can prove at least one Nash equilbirum always exists. These provide the most commonly used definition of a *non cooperative game*.

**Worked Example 1:** We see for the Prisoner's dilemma betray is a dominant strategy for both players (as the game is *symettric*), and so (betray, betray) is the Nash equilibirum. 

**Worked Example 2:** Stag Hunt

Let us consider one more example of a non-cooperative game. In this game, there are two hunters. There are two rabbits and one stag in the hunting range. Before leaving to go hunt, each hunter can only take equipment that catches one type of animal. The stag has more meat than the two rabbits combined, but the hunters have to cooperate with each other to catch it, while the hunter that catches rabbits can catch all the rabbits. The payoff matrix is as follows:

TODO: INSERT STAG HUNT PAYOFF

Here, there is no dominant strategy, and both diagonal entries are Nash equilibria. If hunter A knows that hunter B is going to catch stag, then that hunter should also take equipment to catch the stag. If hunter A knows that hunter B is going to catch the rabbits, then that hunter should go to catch a rabbit too! Once again, self interested agents do not necessarily achieve the optimal outcome, so cooperation is important.

**Definition:** A game is said to be *zero-sum* if the total of gains and losses sum to zero, and thus one player benefits at the direct expense at others. 

**Examples:**
* College admissions. The number of recruited students is fixed, so if someone is sucesfully recruited, some other student is not.
* Chess, tennis, or generally any game where there is one loser and one winner are zero sum.

**Definition:** A game is said to be *positive-sum* if the total of gains and losses is greater than zero. Intuitively, this occurs when resources are somehow increased and there exist approaches, possibly requiring coordination, where desires and needs of all players are satisfied.

**Examples:**
* Trade of goods and services between businesses or nations brings mutually beneficial gains. Both parties benefit from the further specialisation.

**Definition:** The outcome of a game is *Pareto-effiecent* if no player's expected payoff $u_i$ can be increased without some other player's expected payoff $u_j$ decreasing. 

A natural measure of how cooperative some game outcome is is its distance to the *Pareto-frontier*

**Examples:**
* Observe a zero-sum game has all outcomes being *pareto-optimal*.
* For the prisoners dilemma, we see the mutually non cooperative outcome is far from the Pareto-frontier

To define this distance, one needs some ordering on the space of payoffs. 

**Definition:** A social welfare function is a function that ranks outcomes as less desirable, more desirable, or indifferent for every possible pair of outcomes.

One way of doing so is by defining a function $w: \mathbb{R}^n \rightarrow \mathbb{R}$ on the space of payoffs. Choosing $w(u_1,...,u_n)=\sum_i u_i$ gives the utilitarian welfare function, where all players are given equal moral weight. 

## Other classes of problem 

**Definition:** A *collective action problem*, or *free rider problem*, is a problem where there is a cost to for a player to contribute, but all others receive a benefit.

A model for certain classes of problem is as follows. Let $x_j\in\{0,1\}$ be the action of individual $j$, and assume $\beta\in (0,1)$. If $u_j = -x_j + \beta \sum_i x_i$ is the utility of individual $j$, we see $j$ acting reduces their utility, while increaing the utility of the collective. 

**Examples:**
* Carbon emissions. One is not incentivised to reduce their own emissions, but recieves benefit if others do so.

**Definition:** A *common pool resource problem* is one in which agents could deplete resources faster than they can be restored

**Examples:**
* Overfishing. Any individual is incentivised to fish more to increase profits, though if everyone does this, the resource gets completely depleted, reducing everyone's utility.
  
# Cooperative Intelligence

The cooperative intelligence of an agent is its ability to achieve *high joint welfare* in a variety of environments with a *variety of agents*. Why are cooperation problems in the context of AI different?

Let's first consider cooperation in humans. Humans are endowed with a set of dispositions to cooperate. [Further Reading](https://www.practicalethics.ox.ac.uk/uehiro-lectures-2022)
* Disposition to initiate help for strangers
* Disposition to reciprocate
* Disposition to contribute to a shared effort without distinct expectation of return (indirect reciprocity)
* Some intrinsic reward from success at cooperation or collaboration, beyond the actual gain produced
* Some intrinsic interest in whether other have their goals met or are treated fairly
* Disposition to penalize those who are unfair or harmful, even at some expense to oneself

Cooperation can often increase expected utility. This may relate it to morality. The `morality-as-cooperation' theory asserts that "all of human morality is an attempt to solve a cooperative problem" 

TODO: insert cooperation and morality picture

Meanwhile, AI agents are not naturally endowed with these dispostitions, nor with biological features biological creatures use to cooperate. They however can be constructed to have non biological features that may aid cooperation

1. **Understanding** the world, behaviour and preferences of other agents, and dealing with recursive beliefs
2. **Commmunicating** effectively over common ground, dealing with problems of bandwidth and latency, teaching others, and effiecently tackling games with mixed motives.
3. Forming **committments** via devices such as enforcement, automated contracts and arbitration. 
4. Creating trustworthy **institutions**, with good norms and reliable retutation systems.

Cooperative AI research encompasses a wide range of research. To give a flavour of the kinds of research in each subfield, we present a paper for each.

## Understanding
## Communicating
## Committments
## Institutions


# Potential downsides of cooperative AI

As with any technological advance, progress in cooperative AI may be dual to aggrevating certain risks. Downsides with cooperative AI fall can be classified into three board categories.
1. Cooperative competence itself can cause harms, such as by harming those excluded from the colaberative set and underminiming pro-social forms of competition (i.e. **collusion**)
2. Advances in cooperative capabilities may, as a byproduct, **improve coercive capabilities** (e.g., deception).
3.  Successful cooperation often depends on coercion (e.g., pro-social punishment) and competition
(e.g., rivalry as an impetus for improvement), making it hard to pull these apart.
Care should therefore be taken for advances to lead to *differential progress* on cooperation. 
