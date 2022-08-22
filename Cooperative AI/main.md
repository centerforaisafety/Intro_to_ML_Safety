# Cooperative AI
**Contributor(s): Bilal Chughtai**

Cooperative AI is an emerging research area even within safety. As such, this lecture focuses on motivation, background concepts, and promising future directions, over empericial papers.

## Motivation

Problems of cooperation — in which agents have opportunities to improve their joint wellbeing, but are not easily able to do so — are ubiquitous and important. Day to day examples include driving and meeting schedhuling, but the principles scale all the way up to many of the *world's most pressing problems*. Examples include
* Climate change is a problem where everyone contributes, but rational entities may not be incentivised to take action. Coordination is therefore required so that everyone may equally share the burden of action - a **collective action problem**.
* War is a problem where actors need to coordinate to prevent risks of escalation, which results in a worse outcome for both parties. This is an example of a **prisoners dilemma**.
* Curtailing the spread of a deadly pandemic problem requiring the coordination of every individual in a society.

Equally, human civilization and the success of the human species depends on our ability to cooperate. Some of our *greatest successes* can be attributed to cooperation
* Trade allowed individuals to specialise into ....
  
As our societies, economies, and militaries become ever more powerful and connected, the need for cooperation becomes greater. Advanced AI will only exarcerbate this problem, increasing power and interconnectedness further. We've seen examples of this already - the 2010 flash crash was a trillion dollar market crash caused by errors with artificial intelligence powered automated trading systems (though the precise reasons are still not well understood).

Cooperative AI seeks to reverse this problem. It asks *how can we use advances in AI to help us solve cooperation problems?*



## Mathematical Preliminaries

In order to do this, we first require some mathematical framework to think about cooperation. **Game theory** is the study of mathematical models of strategic interactions among rational agents, and encompasses a large range of possible cooperation problems.

**Definition:** A *game* is a triplet $\mathcal{G} = \left(I, (S_i)_{i\in I}, (h_i)_{i\in I}\right)$ where

1. $I$ is the set of players $I=\{1,...,n\}$
2. For each $i\in I$, $S_i$ is player $i$'s set of strategies and $S=\prod_i S_i$
3. For each $i\in I$, $h_i: S\rightarrow \mathbb{R}$ is a function mapping overall strategies to payoffs. One can decompose this into a function $g: S\rightarrow A$ from strategies to outcomes in some outcome space $A$, and then utiltity functions $u_i: A \rightarrow \mathbb{R}$ mapping from outcomes to real numbers. Then $\forall s\in S$, $h_i(s) = u_i (g(s))$.

In words, a game defines a set of possible strategies and payoffs for each agent.

### Non Cooperative Games

**Example:** The Prisoner's Dilemma. 

Suppose two members of a criminal gang are arrested and imprisoned. Each prisoner is in solitary confinement with no means of speaking to or exchanging messages with the other. The police admit they don't have enough evidence to convict the pair on the principal charge. They plan to sentence both to a year in prison on a lesser charge. Simultaneously, the police offer each prisoner a Faustian bargain. The possible outcomes are
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

**Example:** We see for the Prisoner's dilemma betray is a dominant strategy for both players (as the game is *symettric*), and so (betray, betray) is the Nash equilibirum. 

**Example:** Stag Hunt

Let us consider one more example of non-cooperative game. In this game, there are two hunters. There are two rabbits and one stag in the hunting range. Before leaving to go hunt, each hunter can only take equipment that catches one type of animal. The stag has more meat than the two rabbits combined, but the hunters have to cooperate with each other to catch it, while the hunter that catches rabbits can catch all the rabbits. The payoff matrix is as follows:

TODO: INSERT STAG HUNT PAYOFF

Here, there is no dominant strategy, and both diagonal entries are Nash equilibria. If hunter A knows that hunter B is going to catch stag, then that hunter should also take equipment to catch the stag. If hunter A knows that hunter B is going to catch the rabbits, then that hunter should go to catch a rabbit too! Once again, self interested agents do not necessarily achieve the optimal outcome, so cooperation is important.





## Downsides of cooperation