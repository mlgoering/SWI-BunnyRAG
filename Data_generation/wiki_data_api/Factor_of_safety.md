# Factor of safety

In engineering, a factor of safety (FoS) or safety factor (SF) expresses how much stronger a system is than it needs to be for its specified maximum load. Safety factors are often calculated using detailed analysis because comprehensive testing is impractical on many projects, such as bridges and buildings, but the structure's ability to carry a load must be determined to a reasonable accuracy.

Many systems are intentionally built much stronger than needed for normal usage to allow for emergency situations, unexpected loads, misuse, or degradation (reliability).

Margin of safety (MoS or MS) is a related measure, expressed as a relative change.

There are two definitions for the factor of safety (FoS):

The ratio of a structure's absolute strength (structural capability) to actual applied load; this is a measure of the reliability of a particular design. This is a calculated value, and is sometimes referred to, for the sake of clarity, as a realized factor of safety.

A constant required value, imposed by law, standard, specification, contract or custom, to which a structure must conform or exceed. This can be referred to as a design factor, design factor of safety or required factor of safety.

The realized factor of safety must be greater than the required design factor of safety. However, between various industries and engineering groups usage is inconsistent and confusing; there are several definitions used. The cause of much confusion is that various reference books and standards agencies use the factor of safety definitions and terms differently. Building codes, structural and mechanical engineering textbooks often refer to the "factor of safety" as the fraction of total structural capability over what is needed. Those are realized factors of safety (first use). Many undergraduate strength of materials books use "Factor of Safety" as a constant value intended as a minimum target for design (second use).

There are several ways to compare the factor of safety for structures. All the different calculations fundamentally measure the same thing: how much extra load beyond what is intended a structure will actually take (or be required to withstand). The difference between the methods is the way in which the values are calculated and compared. Safety factor values can be thought of as a standardized way for comparing strength and reliability between systems.

The use of a factor of safety does not imply that an item, structure, or design is "safe". Many quality assurance, engineering design, manufacturing, installation, and end-use factors may influence whether or not something is safe in any particular situation.

The difference between the safety factor and design factor (design safety factor) is as follows: The safety factor, or yield stress, is how much the designed part actually will be able to withstand (first usage from above). The design factor, or working stress, is what the item is required to be able to withstand (second usage). The design factor is defined for an application (generally provided in advance and often set by regulatory building codes or policy) and is not an actual calculation, the safety factor is a ratio of maximum strength to intended load for the actual item that was designed.

{\displaystyle {\text{Factor of safety}}={\frac {\text{yield stress}}{\text{working stress}}}}

The design load is the maximum load the part should ever see in service.

By this definition, a structure with an FoS of exactly 1 will support only the design load and no more. Any additional load will cause the structure to fail. A structure with an FoS of 2 will fail at twice the design load.

Many government agencies and industries (such as aerospace) require the use of a margin of safety (MoS or MS) to describe the ratio of the strength of the structure to the requirements. There are two separate definitions for the margin of safety so care is needed to determine which is being used for a given application. One usage of MS is as a measure of capability like FoS. The other usage of MS is as a measure of satisfying design requirements (requirement verification). Margin of safety can be conceptualized (along with the reserve factor explained below) to represent how much of the structure's total capability is held "in reserve" during loading.

MS as a measure of structural capability: This definition of margin of safety commonly seen in textbooks describes what additional load beyond the design load a part can withstand before failing. In effect, this is a measure of excess capability. If the margin is 0, the part will not take any additional load before it fails, if it is negative the part will fail before reaching its design load in service. If the margin is 1, it can withstand one additional load of equal force to the maximum load it was designed to support (i.e. twice the design load).

{\displaystyle {\text{Margin of safety}}={\frac {\text{failure load}}{\text{design load}}}-1}

{\displaystyle {\text{Margin of safety}}={\text{factor of safety}}-1}

MS as a measure of requirement verification: Many agencies and organizations such as NASA and AIAA define the margin of safety including the design factor, in other words, the margin of safety is calculated after applying the design factor. In the case of a margin of 0, the part is at exactly the required strength (the safety factor would equal the design factor). If there is a part with a required design factor of 3 and a margin of 1, the part would have a safety factor of 6 (capable of supporting two loads equal to its design factor of 3, supporting six times the design load before failure). A margin of 0 would mean the part would pass with a safety factor of 3. If the margin is less than 0 in this definition, although the part will not necessarily fail, the design requirement has not been met. A convenience of this usage is that for all applications, a margin of 0 or higher is passing, one does not need to know application details or compare against requirements, just glancing at the margin calculation tells whether the design passes or not. This is helpful for oversight and reviewing on projects with various integrated components, as different components may have various design factors involved and the margin calculation helps prevent confusion.

The design safety factor is provided as a requirement.

{\displaystyle {\text{Margin of safety}}={\frac {\text{failure load}}{\text{design load × design safety factor}}}-1}

{\displaystyle {\text{Margin of safety}}={\frac {\text{realized factor of safety}}{\text{design safety factor}}}-1}

For a successful design, the realized safety factor must always equal or exceed the design safety factor so that the margin of safety is greater than or equal to zero. The margin of safety is sometimes, but infrequently, used as a percentage, i.e., a 0.50 MS is equivalent to a 50% MS. When a design satisfies this test it is said to have a "positive margin", and, conversely, a "negative margin" when it does not.

In the field of nuclear safety (as implemented at US government-owned facilities) the margin of safety has been defined as a quantity that may not be reduced without review by the controlling government office. The US Department of Energy publishes DOE G 424.1-1, "Implementation Guide for Use in Addressing Unreviewed Safety Question Requirements" as a guide for determining how to identify and determine whether a margin of safety will be reduced by a proposed change. The guide develops and applies the concept of a qualitative margin of safety that may not be explicit or quantifiable, yet can be evaluated conceptually to determine whether an increase or decrease will occur with a proposed change. This approach becomes important when examining designs with large or undefined (historical) margins and those that depend on "soft" controls such as programmatic limits or requirements. The commercial US nuclear industry utilized a similar concept in evaluating planned changes until 2001, when 10 CFR 50.59 was revised to capture and apply the information available in facility-specific risk analyses and other quantitative risk management tools.

A measure of strength frequently used in Europe is the reserve factor (RF). With the strength and applied loads expressed in the same units, the reserve factor is defined in one of two ways, depending on the industry:

{\displaystyle {\text{RF}}={\frac {\text{proof strength}}{\text{proof load}}}}

{\displaystyle {\text{RF}}={\frac {\text{ultimate strength}}{\text{ultimate load}}}}

The applied loads have many factors, including factors of safety applied.

For ductile materials (e.g. most metals), it is often required that the factor of safety be checked against both yield and ultimate strengths. The yield calculation will determine the safety factor until the part starts to deform plastically. The ultimate calculation will determine the safety factor until failure. In brittle materials the yield and ultimate strengths are often so close as to be indistinguishable, so it is usually acceptable to only calculate the ultimate safety factor.

Appropriate design factors are based on several considerations, such as the accuracy of predictions on the imposed loads, strength, wear estimates, and the environmental effects to which the product will be exposed in service; the consequences of engineering failure; and the cost of over-engineering the component to achieve that factor of safety . For example, components whose failure could result in substantial financial loss, serious injury, or death may use a safety factor of four or higher (often ten). Non-critical components generally might have a design factor of two. Risk analysis, failure mode and effects analysis, and other tools are commonly used. Design factors for specific applications are often mandated by law, policy, or industry standards.

Buildings commonly use a factor of safety of 2.0 for each structural member. The value for buildings is relatively low because the loads are well understood and most structures are redundant. Pressure vessels use 3.5 to 4.0, automobiles use 3.0, and aircraft and spacecraft use 1.2 to 4.0 depending on the application and materials. Ductile, metallic materials tend to use the lower value while brittle materials use the higher values. The field of aerospace engineering uses generally lower design factors because the costs associated with structural weight are high (i.e. an aircraft with an overall safety factor of 5 would probably be too heavy to get off the ground). This low design factor is why aerospace parts and materials are subject to very stringent quality control and strict preventative maintenance schedules to help ensure reliability. A usually applied Safety Factor is 1.5, but for pressurized fuselage it is 2.0, and for main landing gear structures it is often 1.25.

In some cases it is impractical or impossible for a part to meet the "standard" design factor. The penalties (mass or otherwise) for meeting the requirement would prevent the system from being viable (such as in the case of aircraft or spacecraft). In these cases, it is sometimes determined to allow a component to meet a lower than normal safety factor, often referred to as "waiving" the requirement. Doing this often brings with it extra detailed analysis or quality control verifications to assure the part will perform as desired, as it will be loaded closer to its limits.

For loading that is cyclical, repetitive, or fluctuating, it is important to consider the possibility of metal fatigue when choosing factor of safety. A cyclic load well below a material's yield strength can cause failure if it is repeated through enough cycles.

According to Elishakoff the notion of factor of safety in engineering context was apparently first  introduced  in 1729 by Bernard Forest de Bélidor (1698-1761) who was a French engineer working in hydraulics, mathematics, civil, and military engineering. The philosophical aspects of factors of safety were pursued by Doorn and Hansson.