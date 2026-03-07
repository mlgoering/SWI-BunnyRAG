# Probability theory

Probability theory or probability calculus is the branch of mathematics concerned with probability. Although there are several different probability interpretations, probability theory treats the concept in a rigorous mathematical manner by expressing it through a set of axioms. Typically these axioms formalise probability in terms of a probability space, which assigns a measure taking values between 0 and 1, termed the probability measure, to a set of outcomes called the sample space. Any specified subset of the sample space is called an event.

Central subjects in probability theory include discrete and continuous random variables, probability distributions, and stochastic processes (which provide mathematical abstractions of non-deterministic or uncertain processes or measured quantities that may either be single occurrences or evolve over time in a random fashion).

Although it is not possible to perfectly predict random events, much can be said about their behavior. Two major results in probability theory describing such behaviour are the law of large numbers and the central limit theorem.

As a mathematical foundation for statistics, probability theory is essential to many human activities that involve quantitative analysis of data. Methods of probability theory also apply to descriptions of complex systems given only partial knowledge of their state, as in statistical mechanics or sequential estimation. A great discovery of twentieth-century physics was the probabilistic nature of physical phenomena at atomic scales, described in quantum mechanics.

The modern mathematical theory of probability has its roots in attempts to analyze games of chance by Gerolamo Cardano in the sixteenth century, and by Pierre de Fermat and Blaise Pascal in the seventeenth century (for example the "problem of points"). Christiaan Huygens published a book on the subject in 1657. In the 19th century, what is considered the classical definition of probability was completed by Pierre Laplace.

Initially, probability theory mainly considered discrete events, and its methods were mainly combinatorial. Eventually, analytical considerations compelled the incorporation of continuous variables into the theory.

This culminated in modern probability theory, on foundations laid by Andrey Nikolaevich Kolmogorov. Kolmogorov combined the notion of sample space, introduced by Richard von Mises, and measure theory and presented his axiom system for probability theory in 1933. This became the mostly undisputed axiomatic basis for modern probability theory; but, alternatives exist, such as the adoption of finite rather than countable additivity by Bruno de Finetti.

Most introductions to probability theory treat discrete probability distributions and continuous probability distributions separately. The measure theory-based treatment of probability covers the discrete, continuous, a mix of the two, and more.

Consider an experiment that can produce a number of outcomes. The set of all outcomes is called the sample space of the experiment. The power set of the sample space (or equivalently, the event space) is formed by considering all different collections of possible results. For example, rolling an honest die produces one of six possible results. One collection of possible results corresponds to getting an odd number. Thus, the subset {1,3,5} is an element of the power set of the sample space of dice rolls. These collections are called events. In this case, {1,3,5} is the event that the die falls on some odd number. If the results that actually occur fall in a given event, that event is said to have occurred.

Probability is a way of assigning every "event" a value between zero and one, with the requirement that the event made up of all possible results (in our example, the event {1,2,3,4,5,6}) be assigned a value of one. To qualify as a probability distribution, the assignment of values must satisfy the requirement that if you look at a collection of mutually exclusive events (events that contain no common results, e.g., the events {1,6}, {3}, and {2,4} are all mutually exclusive), the probability that any of these events occurs is given by the sum of the probabilities of the events.

The probability that any one of the events {1,6}, {3}, or {2,4} will occur is 5/6. This is the same as saying that the probability of event {1,2,3,4,6} is 5/6. This event encompasses the possibility of any number except five being rolled. The mutually exclusive event {5} has a probability of 1/6, and the event {1,2,3,4,5,6} has a probability of 1, that is, absolute certainty.

When doing calculations using the outcomes of an experiment, it is necessary that all those elementary events have a number assigned to them. This is done using a random variable. A random variable is a function that assigns to each elementary event in the sample space a real number. This function is usually denoted by a capital letter. In the case of a die, the assignment of a number to certain elementary events can be done using the identity function. This does not always work. For example, when flipping a coin the two possible outcomes are "heads" and "tails". In this example, the random variable X could assign to the outcome "heads" the number "0" (

Discrete probability theory deals with events that occur in countable sample spaces.

Examples: Throwing dice, experiments with decks of cards, random walk, and tossing coins.

Initially the probability of an event to occur was defined as the number of cases favorable for the event, over the number of total outcomes possible in an equiprobable sample space: see Classical definition of probability.

For example, if the event is "occurrence of an even number when a dice is rolled", the probability is given by

, since 3 faces out of the 6 have even numbers and each face has the same probability of appearing.

The modern definition starts with a finite or countable set called the sample space, which relates to the set of all possible outcomes in classical sense, denoted by

is attached, which satisfies the following properties:

{\displaystyle f(x)\in [0,1]{\mbox{ for all }}x\in \Omega \,;}

That is, the probability function f(x) lies between zero and one for every value of x in the sample space Ω, and the sum of f(x) over all values x in the sample space Ω is equal to 1. An event is defined as any subset

{\displaystyle \mathbb {P} (E)=\sum _{x\in E}f(x)\,.}

So, the probability of the entire sample space is 1, and the probability of the null event is 0.

mapping a point in the sample space to the "probability" value is called a probability mass function abbreviated as pmf.

Continuous probability theory deals with events that occur in a continuous sample space.

The classical definition breaks down when confronted with the continuous case. See Bertrand's paradox.

If the sample space of a random variable X is the set of real numbers (

) or a subset thereof, then a function called the cumulative distribution function (CDF)

. That is, F(x) returns the probability that X will be less than or equal to x.

The CDF necessarily satisfies the following properties.

is a monotonically non-decreasing, right-continuous function;

{\displaystyle \lim _{x\rightarrow -\infty }F(x)=0\,;}

{\displaystyle \lim _{x\rightarrow \infty }F(x)=1\,.}

is said to have a continuous probability distribution if the corresponding CDF

is absolutely continuous, then its derivative exists almost everywhere and integrating the derivative gives us the CDF back again. In this case, the random variable X is said to have a probability density function (PDF) or simply density

, the probability of the random variable X being in

{\displaystyle \mathbb {P} (X\in E)=\int _{x\in E}dF(x)\,.}

{\displaystyle \mathbb {P} (X\in E)=\int _{x\in E}f(x)\,dx\,.}

Whereas the PDF exists only for continuous random variables, the CDF exists for all random variables (including discrete random variables) that take values in

These concepts can be generalized for multidimensional cases on

The utility of the measure-theoretic treatment of probability is that it unifies the discrete and the continuous cases, and makes the difference a question of which measure is used. Furthermore, it covers distributions that are neither discrete nor continuous nor mixtures of the two.

An example of such distributions could be a mix of discrete and continuous distributions—for example, a random variable that is 0 with probability 1/2, and takes a random value from a normal distribution with probability 1/2. It can still be studied to some extent by considering it to have a PDF of

Other distributions may not even be a mix, for example, the Cantor distribution has no positive probability for any single point, neither does it have a density. The modern approach to probability theory solves these problems using measure theory to define the probability space:

is the Borel σ-algebra on the set of real numbers, then there is a unique probability measure on

for any CDF, and vice versa. The measure corresponding to a CDF is said to be induced by the CDF. This measure coincides with the pmf for discrete variables and PDF for continuous variables, making the measure-theoretic approach free of fallacies.

{\displaystyle \mathbb {P} (E)=\int _{\omega \in E}\mu _{F}(d\omega )\,}

where the integration is with respect to the measure

Along with providing better understanding and unification of discrete and continuous probabilities, measure-theoretic treatment also allows us to work on probabilities outside

, as in the theory of stochastic processes. For example, to study Brownian motion, probability is defined on a space of functions.

When it is convenient to work with a dominating measure, the Radon–Nikodym theorem is used to define a density as the Radon–Nikodym derivative of the probability distribution of interest with respect to this dominating measure.  Discrete densities are usually defined as this derivative with respect to a counting measure over the set of all possible outcomes.  Densities for absolutely continuous distributions are usually defined as this derivative with respect to the Lebesgue measure.  If a theorem can be proved in this general setting, it holds for both discrete and continuous distributions as well as others;  separate proofs are not required for discrete and continuous distributions.

Certain random variables occur very often in probability theory because they well describe many natural or physical processes. Their distributions, therefore, have gained special importance in probability theory. Some fundamental discrete distributions are the discrete uniform, Bernoulli, binomial, negative binomial, Poisson and geometric distributions. Important continuous distributions include the continuous uniform, normal, exponential, gamma and beta distributions.

In probability theory, there are several notions of convergence for random variables. They are listed below in the order of strength, i.e., any subsequent notion of convergence in the list implies convergence according to all of the preceding notions.

is continuous. Weak convergence is also called convergence in distribution.

{\displaystyle \displaystyle X_{n}\,{\xrightarrow {\mathcal {D}}}\,X}

{\displaystyle \lim _{n\rightarrow \infty }\mathbb {P} \left(\left|X_{n}-X\right|\geq \varepsilon \right)=0}

{\displaystyle \displaystyle X_{n}\,\xrightarrow {\mathbb {P} } \,X}

{\displaystyle \mathbb {P} (\lim _{n\rightarrow \infty }X_{n}=X)=1}

. Strong convergence is also known as almost sure convergence.

{\displaystyle \displaystyle X_{n}\,{\xrightarrow {\mathrm {a.s.} }}\,X}

As the names indicate, weak convergence is weaker than strong convergence. In fact, strong convergence implies convergence in probability, and convergence in probability implies weak convergence. The reverse statements are not always true.

Common intuition suggests that if a fair coin is tossed many times, then roughly half of the time it will turn up heads, and the other half it will turn up tails. Furthermore, the more often the coin is tossed, the more likely it should be that the ratio of the number of heads to the number of tails will approach unity. Modern probability theory provides a formal version of this intuitive idea, known as the law of large numbers. This law is remarkable because it is not assumed in the foundations of probability theory, but instead emerges from these foundations as a theorem. Since it links theoretically derived probabilities to their actual frequency of occurrence in the real world, the law of large numbers is considered as a pillar in the history of statistical theory and has had widespread influence.

The law of large numbers (LLN) states that the sample average

{\displaystyle {\overline {X}}_{n}={\frac {1}{n}}{\sum _{k=1}^{n}X_{k}}}

of a sequence of independent and identically distributed random variables

converges towards their common expectation (expected value)

It is in the different forms of convergence of random variables that separates the weak and the strong law of large numbers

{\displaystyle \displaystyle {\overline {X}}_{n}\,\xrightarrow {\mathbb {P} } \,\mu }

{\displaystyle \displaystyle {\overline {X}}_{n}\,{\xrightarrow {\mathrm {a.\,s.} }}\,\mu }

It follows from the LLN that if an event of probability p is observed repeatedly during independent experiments, the ratio of the observed frequency of that event to the total number of repetitions converges towards p.

are independent Bernoulli random variables taking values 1 with probability p and 0 with probability 1-p, then

The central limit theorem (CLT) explains the ubiquitous occurrence of the normal distribution in nature, and this theorem, according to David Williams, "is one of the great results of mathematics."

The theorem states that the average of many independent and identically distributed random variables with finite variance tends towards a normal distribution irrespective of the distribution followed by the original random variables. Formally, let

{\displaystyle Z_{n}={\frac {\sum _{i=1}^{n}(X_{i}-\mu )}{\sigma {\sqrt {n}}}}\,}

converges in distribution to a standard normal random variable.

For some classes of random variables, the classic central limit theorem works rather fast, as illustrated in the Berry–Esseen theorem. For example, the distributions with finite  first, second, and third moment from the exponential family; on the other hand, for some random variables of the heavy tail and fat tail variety, it works very slowly or may not work at all: in such cases one may use the Generalized Central Limit Theorem (GCLT).