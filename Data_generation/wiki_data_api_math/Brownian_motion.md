# Brownian motion

Brownian motion is the random motion of particles suspended in a medium (a liquid or a gas).  The traditional mathematical formulation of Brownian motion is that of the Wiener process, which is often called Brownian motion, even in mathematical sources.

This motion pattern typically consists of random fluctuations in a particle's position inside a fluid sub-domain, followed by a relocation to another sub-domain. Each relocation is followed by more fluctuations within the new closed volume. This pattern describes a fluid at thermal equilibrium, defined by a given temperature. Within such a fluid, there exists no preferential direction of flow (as in transport phenomena). More specifically, the fluid's overall linear and angular momenta remain null over time. The kinetic energies of the molecular Brownian motions, together with those of molecular rotations and vibrations, sum up to the caloric component of a fluid's internal energy (the equipartition theorem).

This motion is named after the Scottish botanist Robert Brown, who first described the phenomenon in 1827, while looking through a microscope at pollen of the plant Clarkia pulchella immersed in water. In 1900, the French mathematician Louis Bachelier modeled the stochastic process now called Brownian motion in his doctoral thesis, The Theory of Speculation (Théorie de la spéculation), prepared under the supervision of Henri Poincaré. Then, in 1905, theoretical physicist Albert Einstein published a paper in which he modelled the motion of the pollen particles as being moved by individual water molecules, making one of his first major scientific contributions.

The direction of the force of atomic bombardment is constantly changing, and at different times the particle is hit more on one side than another, leading to the seemingly random nature of the motion. This explanation of Brownian motion served as convincing evidence that atoms and molecules exist and was further verified experimentally by Jean Perrin in 1908. Perrin was awarded the Nobel Prize in Physics in 1926 "for his work on the discontinuous structure of matter".

The many-body interactions that yield the Brownian pattern cannot be solved by a model accounting for every involved molecule. Consequently, only probabilistic models applied to molecular populations can be employed to describe it. Two such models of the statistical mechanics, due to Einstein and Smoluchowski, are presented below. Another, pure probabilistic class of models is the class of the stochastic process models. There exist sequences of both simpler and more complicated stochastic processes which converge (in the limit) to Brownian motion (see random walk and Donsker's theorem).

The Roman philosopher-poet Lucretius' scientific poem On the Nature of Things (c. 60 BC) has a remarkable description of the motion of dust particles in verses 113–140 from Book II. He uses this as a proof of the existence of atoms:

Observe what happens when sunbeams are admitted into a building and shed light on its shadowy places. You will see a multitude of tiny particles mingling in a multitude of ways... their dancing is an actual indication of underlying movements of matter that are hidden from our sight... It originates with the atoms which move of themselves [i.e., spontaneously]. Then those small compound bodies that are least removed from the impetus of the atoms are set in motion by the impact of their invisible blows and in turn cannon against slightly larger bodies. So the movement mounts up from the atoms and gradually emerges to the level of our senses so that those bodies are in motion that we see in sunbeams, moved by blows that remain invisible.

Although the mingling, tumbling motion of dust particles is caused largely by air currents, the glittering, jiggling motion of small dust particles is caused chiefly by true Brownian dynamics; Lucretius "perfectly describes and explains the Brownian movement by a wrong example".

The discovery of this phenomenon is credited to the botanist Robert Brown in 1827. Brown was studying plant reproduction when he observed pollen grains of the plant Clarkia pulchella  in water under a microscope. These grains contain minute particles on the order of 1/4000th of an inch (6 microns) in size. He observed these particles executing a jittery motion. By repeating the experiment with particles of inorganic matter he was able to rule out that the motion was life-related, although its origin was yet to be explained.

The mathematics of much of stochastic analysis including the mathematics of Brownian motion was introduced by Louis Bachelier in 1900 in his PhD thesis "The theory of speculation", in which he presented an analysis of the stock and option markets. However this work was largely unknown until the 1950s.

Albert Einstein (in one of his 1905 papers) provided an explanation of Brownian motion in terms of atoms and molecules at a time when their existence was still debated. Einstein proved the relation between the probability distribution of a Brownian particle and the diffusion equation. These equations describing Brownian motion were subsequently verified by the experimental work of Jean Baptiste Perrin in 1908, leading to his Nobel prize. Norbert Wiener gave the first complete and rigorous mathematical analysis in 1923, leading to the underlying mathematical concept being called a Wiener process.

The instantaneous velocity of the Brownian motion can be defined as v = Δx/Δt, when Δt << τ, where τ is the momentum relaxation time.

In 2010, the instantaneous velocity of a Brownian particle (a glass microsphere trapped in air with optical tweezers) was measured successfully. The velocity data verified the Maxwell–Boltzmann velocity distribution, and the equipartition theorem for a Brownian particle.

There are two parts to Einstein's theory: the first part consists in the formulation of a diffusion equation for Brownian particles, in which the diffusion coefficient is related to the mean squared displacement of a Brownian particle, while the second part consists in relating the diffusion coefficient to measurable physical quantities. In this way Einstein was able to determine the size of atoms, and how many atoms there are in a mole, or the molecular weight in grams, of a gas. In accordance to Avogadro's law, this volume is the same for all ideal gases, which is 22.414 liters at standard temperature and pressure. The number of atoms contained in this volume is referred to as the Avogadro number, and the determination of this number is tantamount to the knowledge of the mass of an atom, since the latter is obtained by dividing the molar mass of the gas by the Avogadro constant.

The first part of Einstein's argument was to determine how far a Brownian particle travels in a given time interval. Classical mechanics is unable to determine this distance because of the enormous number of bombardments a Brownian particle will undergo, roughly of the order of 1014 collisions per second.

He regarded the increment of particle positions in time

in a one-dimensional (x) space (with the coordinates chosen so that the origin lies at the initial position of the particle) as a random variable (

, i.e., the probability density of the particle incrementing its position from

). Further, assuming conservation of particle number, he expanded the number density

{\displaystyle {\begin{aligned}\rho (x,t+\tau )={}&\rho (x,t)+\tau {\frac {\partial \rho (x,t)}{\partial t}}+\cdots \\[2ex]={}&\int _{-\infty }^{\infty }\rho (x-q,t)\,\varphi (q)\,dq=\mathbb {E} _{q}{\left[\rho (x-q,t)\right]}\\[1ex]={}&\rho (x,t)\,\int _{-\infty }^{\infty }\varphi (q)\,dq-{\frac {\partial \rho }{\partial x}}\,\int _{-\infty }^{\infty }q\,\varphi (q)\,dq+{\frac {\partial ^{2}\rho }{\partial x^{2}}}\,\int _{-\infty }^{\infty }{\frac {q^{2}}{2}}\varphi (q)\,dq+\cdots \\[1ex]={}&\rho (x,t)\cdot 1-0+{\cfrac {\partial ^{2}\rho }{\partial x^{2}}}\,\int _{-\infty }^{\infty }{\frac {q^{2}}{2}}\varphi (q)\,dq+\cdots \end{aligned}}}

. The integral in the first term is equal to one by the definition of probability, and the second and other even terms (i.e. first and other odd moments) vanish because of space symmetry. What is left gives rise to the following relation:

{\displaystyle {\frac {\partial \rho }{\partial t}}={\frac {\partial ^{2}\rho }{\partial x^{2}}}\cdot \int _{-\infty }^{\infty }{\frac {q^{2}}{2\tau }}\varphi (q)\,dq+{\text{higher-order even moments.}}}

Where the coefficient after the Laplacian, the second moment of probability of displacement

{\displaystyle D=\int _{-\infty }^{\infty }{\frac {q^{2}}{2\tau }}\varphi (q)\,dq.}

Then the density of Brownian particles ρ at point x at time t satisfies the diffusion equation:

{\displaystyle {\frac {\partial \rho }{\partial t}}=D\cdot {\frac {\partial ^{2}\rho }{\partial x^{2}}},}

Assuming that N particles start from the origin at the initial time t = 0, the diffusion equation has the solution:

{\displaystyle \rho (x,t)={\frac {N}{\sqrt {4\pi Dt}}}\exp {\left(-{\frac {x^{2}}{4Dt}}\right)}.}

This expression (which is a normal distribution with the mean

) allowed Einstein to calculate the moments directly. The first moment is seen to vanish, meaning that the Brownian particle is equally likely to move to the left as it is to move to the right. The second moment is, however, non-vanishing, being given by

{\displaystyle \mathbb {E} {\left[x^{2}\right]}=2Dt.}

This equation expresses the mean squared displacement in terms of the time elapsed and the diffusivity. From this expression Einstein argued that the displacement of a Brownian particle is not proportional to the elapsed time, but rather to its square root. His argument is based on a conceptual switch from the "ensemble" of Brownian particles to the "single" Brownian particle: we can speak of the relative number of particles at a single instant just as well as of the time it takes a Brownian particle to reach a given point.

The second part of Einstein's theory relates the diffusion constant to physically measurable quantities, such as the mean squared displacement of a particle in a given time interval. This result enables the experimental determination of the Avogadro number and therefore the size of molecules. Einstein analyzed a dynamic equilibrium being established between opposing forces. The beauty of his argument is that the final result does not depend upon which forces are involved in setting up the dynamic equilibrium.

In his original treatment, Einstein considered an osmotic pressure experiment, but the same conclusion can be reached in other ways.

Consider, for instance, particles suspended in a viscous fluid in a gravitational field. Gravity tends to make the particles settle, whereas diffusion acts to homogenize them, driving them into regions of smaller concentration. Under the action of gravity, a particle acquires a downward speed of v = μmg, where m is the mass of the particle, g is the acceleration due to gravity, and μ is the particle's mobility in the fluid. George Stokes had shown that the mobility for a spherical particle with radius r is

, where η is the dynamic viscosity of the fluid. In a state of dynamic equilibrium, and under the hypothesis of isothermal fluid, the particles are distributed according to the barometric distribution

{\displaystyle \rho =\rho _{o}\,\exp \left({-{\frac {mgh}{k_{\text{B}}T}}}\right),}

where ρ − ρo is the difference in density of particles separated by a height difference, of

, kB is the Boltzmann constant (the ratio of the universal gas constant, R, to the Avogadro constant, NA), and T is the absolute temperature.

Dynamic equilibrium is established because the more that particles are pulled down by gravity, the greater the tendency for the particles to migrate to regions of lower concentration. The flux is given by Fick's law,

where J = ρv. Introducing the formula for ρ, we find that

In a state of dynamical equilibrium, this speed must also be equal to v = μmg. Both expressions for v are proportional to mg, reflecting that the derivation is independent of the type of forces considered. Similarly, one can derive an equivalent formula for identical charged particles of charge q in a uniform electric field of magnitude E, where mg is replaced with the electrostatic force qE. Equating these two expressions yields the Einstein relation for the diffusivity, independent of mg or qE or other such forces:

{\displaystyle {\frac {\mathbb {E} {\left[x^{2}\right]}}{2t}}=D=\mu k_{\text{B}}T={\frac {\mu RT}{N_{\text{A}}}}={\frac {RT}{6\pi \eta rN_{\text{A}}}}.}

Here the first equality follows from the first part of Einstein's theory, the third equality follows from the definition of the Boltzmann constant as kB = R / NA, and the fourth equality follows from Stokes's formula for the mobility. By measuring the mean squared displacement over a time interval along with the universal gas constant R, the temperature T, the viscosity η, and the particle radius r, the Avogadro constant NA can be determined.

The type of dynamical equilibrium proposed by Einstein was not new. It had been pointed out previously by J. J. Thomson in his series of lectures at Yale University in May 1903 that the dynamic equilibrium between the velocity generated by a concentration gradient given by Fick's law and the velocity due to the variation of the partial pressure caused when ions are set in motion "gives us a method of determining Avogadro's constant which is independent of any hypothesis as to the shape or size of molecules, or of the way in which they act upon each other".

An identical expression to Einstein's formula for the diffusion coefficient was also found by Walther Nernst in 1888 in which he expressed the diffusion coefficient as the ratio of the osmotic pressure to the ratio of the frictional force and the velocity to which it gives rise. The former was equated to the law of van 't Hoff while the latter was given by Stokes's law. He writes

is the osmotic pressure and k is the ratio of the frictional force to the molecular viscosity which he assumes is given by Stokes's formula for the viscosity. Introducing the ideal gas law per unit volume for the osmotic pressure, the formula becomes identical to that of Einstein's. The use of Stokes's law in Nernst's case, as well as in Einstein and Smoluchowski, is not strictly applicable since it does not apply to the case where the radius of the sphere is small in comparison with the mean free path.

Confirming Einstein's formula experimentally proved difficult.

Initial attempts by Theodor Svedberg in 1906 and 1907 were critiqued by Einstein and by Perrin as not measuring a quantity directly comparable to the formula. Victor Henri in 1908 took cinematographic shots through a microscope and found quantitative disagreement with the formula but again the analysis was uncertain. Einstein's predictions were finally confirmed in a series of experiments carried out by Chaudesaigues in 1908 and Perrin in 1909. The confirmation of Einstein's theory constituted empirical progress for the kinetic theory of heat. In essence, Einstein showed that the motion can be predicted directly from the kinetic model of thermal equilibrium. The importance of the theory lay in the fact that it confirmed the kinetic theory's account of the second law of thermodynamics as being an essentially statistical law.

Smoluchowski's theory of Brownian motion starts from the same premise as that of Einstein and derives the same probability distribution ρ(x, t) for the displacement of a Brownian particle along the x in time t. He therefore gets the same expression for the mean squared displacement:

{\displaystyle \mathbb {E} {\left[(\Delta x)^{2}\right]}}

. However, when he relates it to a particle of mass m moving at a velocity u which is the result of a frictional force governed by Stokes's law, he finds

{\displaystyle \mathbb {E} {\left[(\Delta x)^{2}\right]}=2Dt=t{\frac {32}{81}}{\frac {mu^{2}}{\pi \mu a}}=t{\frac {64}{27}}{\frac {{\frac {1}{2}}mu^{2}}{3\pi \mu a}},}

where μ is the viscosity coefficient, and a is the radius of the particle. Associating the kinetic energy

with the thermal energy RT/N, the expression for the mean squared displacement is 64/27 times that found by Einstein. The fraction 27/64 was commented on by Arnold Sommerfeld in his necrology on Smoluchowski: "The numerical coefficient of Einstein, which differs from Smoluchowski by 27/64 can only be put in doubt."

Smoluchowski attempts to answer the question of why a Brownian particle should be displaced by bombardments of smaller particles when the probabilities for striking it in the forward and rear directions are equal.

If the probability of m gains and n − m losses follows a binomial distribution,

with equal a priori probabilities of 1/2, the mean total gain is

{\displaystyle \mathbb {E} {\left[2m-n\right]}=\sum _{m={\frac {n}{2}}}^{n}(2m-n)P_{m,n}={\frac {nn!}{2^{n+1}\left[\left({\frac {n}{2}}\right)!\right]^{2}}}.}

If n is large enough so that Stirling's approximation can be used in the form

{\displaystyle n!\approx \left({\frac {n}{e}}\right)^{n}{\sqrt {2\pi n}},}

{\displaystyle \mathbb {E} {\left[2m-n\right]}\approx {\sqrt {\frac {n}{2\pi }}},}

showing that it increases as the square root of the total population.

Suppose that a Brownian particle of mass M is surrounded by lighter particles of mass m which are traveling at a speed u. Then, reasons Smoluchowski, in any collision between a surrounding and Brownian particles, the velocity transmitted to the latter will be mu/M. This ratio is of the order of 10−7 cm/s. But we also have to take into consideration that in a gas there will be more than 1016 collisions in a second, and even greater in a liquid where we expect that there will be 1020 collision in one second. Some of these collisions will tend to accelerate the Brownian particle; others will tend to decelerate it. If there is a mean excess of one kind of collision or the other to be of the order of 108 to 1010 collisions in one second, then velocity of the Brownian particle may be anywhere between 10–1000 cm/s. Thus, even though there are equal probabilities for forward and backward collisions there will be a net tendency to keep the Brownian particle in motion, just as the ballot theorem predicts.

These orders of magnitude are not exact because they do not take into consideration the velocity of the Brownian particle, U, which depends on the collisions that tend to accelerate and decelerate it. The larger U is, the greater will be the collisions that will retard it so that the velocity of a Brownian particle can never increase without limit. Could such a process occur, it would be tantamount to a perpetual motion of the second type. And since equipartition of energy applies, the kinetic energy of the Brownian particle,

, will be equal, on the average, to the kinetic energy of the surrounding fluid particle,

In 1906, Smoluchowski published a one-dimensional model to describe a particle undergoing Brownian motion. The model assumes collisions with M ≫ m where M is the test particle's mass and m the mass of one of the individual particles composing the fluid. It is assumed that the particle collisions are confined to one dimension and that it is equally probable for the test particle to be hit from the left as from the right. It is also assumed that every collision always imparts the same magnitude of ΔV. If NR is the number of collisions from the right and NL the number of collisions from the left then after N collisions the particle's velocity will have changed by ΔV(2NR − N). The multiplicity is then simply given by:

{\displaystyle {\binom {N}{N_{\text{R}}}}={\frac {N!}{N_{\text{R}}!(N-N_{\text{R}})!}}}

and the total number of possible states is given by 2N. Therefore, the probability of the particle being hit from the right NR times is:

{\displaystyle P_{N}(N_{\text{R}})={\frac {N!}{2^{N}N_{\text{R}}!(N-N_{\text{R}})!}}}

As a result of its simplicity, Smoluchowski's 1D model can only qualitatively describe Brownian motion. For a realistic particle undergoing Brownian motion in a fluid, many of the assumptions don't apply. For example, the assumption that on average there are an equal number of collisions from the right as from the left falls apart once the particle is in motion. Also, there would be a distribution of different possible ΔVs instead of always just one in a realistic situation.

The diffusion equation yields an approximation of the time evolution of the probability density function associated with the position of the particle going under a Brownian movement under the physical definition. The approximation becomes valid on timescales much larger than the timescale of individual atomic collisions, since it does not include a term to describe the acceleration of particles during collision. The time evolution of the position of the Brownian particle over all time scales described using the Langevin equation, an equation that involves a random force field representing the effect of the thermal fluctuations of the solvent on the particle. At longer times scales, where acceleration is negligible, individual particle dynamics can be approximated using Brownian dynamics in place of Langevin dynamics.

In stellar dynamics, a massive body (star, black hole, etc.) can experience Brownian motion as it responds to gravitational forces from surrounding stars. The rms velocity V of the massive object, of mass M, is related to the rms velocity

is the mass of the background stars. The gravitational force from the massive object causes nearby stars to move faster than they otherwise would, increasing both

and V. The Brownian velocity of Sgr A*, the supermassive black hole at the center of the Milky Way galaxy, is predicted from this formula to be less than 1 km s−1.

In mathematics, Brownian motion is described by the Wiener process, a continuous-time stochastic process named in honor of Norbert Wiener. It is one of the best known Lévy processes (càdlàg stochastic processes with stationary independent increments) and occurs frequently in pure and applied mathematics, economics and physics.

The Wiener process Wt is characterized by four facts:

{\displaystyle W_{t}-W_{s}\sim {\mathcal {N}}(0,t-s)}

denotes the normal distribution with expected value μ and variance σ2. The condition that it has independent increments means that if

are independent random variables. In addition, for some filtration

An alternative characterisation of the Wiener process is the so-called Lévy characterisation that says that the Wiener process is an almost surely continuous martingale with W0 = 0 and quadratic variation

A third characterisation is that the Wiener process has a spectral representation as a sine series whose coefficients are independent

random variables. This representation can be obtained using the Kosambi–Karhunen–Loève theorem.

The Wiener process can be constructed as the scaling limit of a random walk, or other discrete-time stochastic processes with stationary independent increments. This is known as Donsker's theorem. Like the random walk, the Wiener process is recurrent in one or two dimensions (meaning that it returns almost surely to any fixed neighborhood of the origin infinitely often) whereas it is not recurrent in dimensions three and higher. Unlike the random walk, it is scale invariant.

A d-dimensional Gaussian free field has been described as "a d-dimensional-time analog of Brownian motion."

The Brownian motion can be modeled by a random walk.

In the general case, Brownian motion is a Markov process and described by stochastic integral equations.

The French mathematician Paul Lévy proved the following theorem, which gives a necessary and sufficient condition for a continuous Rn-valued stochastic process X to actually be n-dimensional Brownian motion. Hence, Lévy's condition can actually be used as an alternative definition of Brownian motion.

Let X = (X1, ..., Xn) be a continuous stochastic process on a probability space (Ω, Σ, P) taking values in Rn. Then the following are equivalent:

X is a Brownian motion with respect to P, i.e., the law of X with respect to P is the same as the law of an n-dimensional Brownian motion, i.e., the push-forward measure X∗(P) is classical Wiener measure on C0([0, ∞); Rn).

X is a martingale with respect to P (and its own natural filtration); and

for all 1 ≤ i, j ≤ n, Xi(t) Xj(t) − δij t is a martingale with respect to P (and its own natural filtration), where δij denotes the Kronecker delta.

can be found from the power spectral density, formally defined as

{\displaystyle S(\omega )=\lim _{T\to \infty }{\frac {1}{T}}\mathbb {E} \left\{\left|\int _{0}^{T}e^{i\omega t}X_{t}dt\right|^{2}\right\},}

stands for the expected value. The power spectral density of Brownian motion is found to be

{\displaystyle S_{BM}(\omega )={\frac {4D}{\omega ^{2}}}.}

where D is the diffusion coefficient of Xt. For naturally occurring signals, the spectral content can be found from the power spectral density of a single realization, with finite available time, i.e.,

{\displaystyle S^{(1)}(\omega ,T)={\frac {1}{T}}\left|\int _{0}^{T}e^{i\omega t}X_{t}dt\right|^{2},}

which for an individual realization of a Brownian motion trajectory, it is found to have expected value

{\displaystyle \mu _{\text{BM}}(\omega ,T)={\frac {4D}{\omega ^{2}}}\left[1-{\frac {\sin \left(\omega T\right)}{\omega T}}\right]}

{\displaystyle \sigma _{S}^{2}(f,T)=\mathbb {E} \left\{\left(S_{T}^{(j)}(f)\right)^{2}\right\}-\mu _{S}^{2}(f,T)={\frac {20D^{2}}{f^{4}}}\left[1-{\Big (}6-\cos \left(fT\right){\Big )}{\frac {2\sin \left(fT\right)}{5fT}}+{\frac {{\Big (}17-\cos \left(2fT\right)-16\cos \left(fT\right){\Big )}}{10f^{2}T^{2}}}\right].}

For sufficiently long realization times, the expected value of the power spectrum of a single trajectory converges to the formally defined power spectral density

Brownian motion is usually considered to take place in Euclidean space. It is natural to consider how such motion generalizes to more complex shapes, such as surfaces or higher dimensional manifolds. The formalization requires the space to possess some form of a derivative, as well as a metric, so that a Laplacian can be defined. Both of these are available on Riemannian manifolds.

Riemannian manifolds have the property that geodesics can be described in polar coordinates; that is, displacements are always in a radial direction, at some given angle. Uniform random motion is then described by Gaussians along the radial direction, independent of the angle, the same as in Euclidean space.

The infinitesimal generator (and hence characteristic operator) of Brownian motion on Euclidean Rn is ⁠1/2⁠Δ, where Δ denotes the Laplace operator. Brownian motion on an m-dimensional Riemannian manifold (M, g) can be defined as diffusion on M with the characteristic operator given by ⁠1/2⁠ΔLB, half the Laplace–Beltrami operator ΔLB.

One of the topics of study is a characterization of the Poincaré recurrence time for such systems.

The narrow escape problem is a ubiquitous problem in biology, biophysics and cellular biology which has the following formulation: a Brownian particle (ion, molecule, or protein) is confined to a bounded domain (a compartment or a cell) by a reflecting boundary, except for a small window through which it can escape. The narrow escape problem is that of calculating the mean escape time. This time diverges as the window shrinks, thus rendering the calculation a singular perturbation problem.