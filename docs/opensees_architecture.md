# OpenSees Architecture Reference

Detailed C++ architecture of OpenSees, extracted from the source in `OPENSEES SRC/`. Use this when implementing or extending oneFEM — the goal is to faithfully mirror OpenSees patterns in Python.

---

## 1. Top-Level Directory Structure

```
OPENSEES SRC/
├── domain/           # Model definition (Domain, Node, Element, Constraint, LoadPattern)
├── analysis/         # Solution framework (Analysis, Algorithm, Integrator, Model)
├── element/          # Element implementations (truss, beam, shell, solid, zerolength)
├── material/         # Constitutive models (uniaxial, nD, section)
├── matrix/           # Linear algebra primitives (Vector, Matrix, ID)
├── system_of_eqn/    # Equation solvers (dense, banded, sparse)
├── convergenceTest/  # Iteration convergence criteria
├── handler/          # Constraint enforcement (Plain, Penalty, Lagrange, Transformation)
├── graph/            # Connectivity graphs and numbering algorithms
├── coordTransformation/  # Local-to-global element transforms
├── recorder/         # Output recording (node, element, envelope)
├── tagged/           # TaggedObject base + storage containers
├── actor/            # Parallel computing & serialization (Channel, MovableObject)
├── modelbuilder/     # Tcl/Python model construction helpers
├── reliability/      # Reliability analysis extensions
├── damage/           # Damage models
├── damping/          # Damping models
├── database/         # State persistence (file, MySQL, etc.)
├── interpreter/      # Tcl command bindings
├── runtime/          # Runtime context and globals
└── utility/          # Timers, file utilities
```

---

## 2. Core Design Patterns

### 2.1 Strategy Pattern (Pluggable Components)

Analysis behavior is assembled from interchangeable strategies:

```cpp
StaticAnalysis analysis(
    domain,              // Domain
    handler,             // ConstraintHandler  (Plain, Penalty, Lagrange, Transformation)
    numberer,            // DOF_Numberer       (Plain, RCM, AMD)
    model,               // AnalysisModel      (bridge between Domain and solvers)
    algorithm,           // EquiSolnAlgo       (Linear, Newton, BFGS, KrylovNewton)
    soe,                 // LinearSOE          (FullGeneral, ProfileSPD, UmfPack, SuperLU)
    integrator,          // StaticIntegrator   (LoadControl, DisplacementControl, ArcLength)
    test                 // ConvergenceTest    (NormUnbalance, NormDispIncr, EnergyIncr)
);
```

Any component can be swapped at runtime via `analysis.setAlgorithm(...)`, etc.

### 2.2 Trial/Commit Two-State Pattern

**Every stateful object** (Node, Material, Element) maintains two independent state copies:
- **Trial**: current iteration, tentative, may be reverted
- **Committed**: last converged step, permanent, safe fallback

```cpp
// Interface (present on Node, Element, Material):
int commitState();           // trial → committed (deep copy!)
int revertToLastCommit();    // committed → trial (deep copy!)
int revertToStart();         // reset to initial state
```

**Critical**: commit/revert must deep-copy, never alias. This pattern enables:
- Newton iteration with safe rollback on divergence
- Multiscale FE² nesting (each scale independently manages convergence)

### 2.3 TaggedObject and Storage

All domain objects inherit from `TaggedObject` (integer tag for O(1) lookup):

```cpp
class TaggedObject {
    int theTag;
public:
    int getTag() const { return theTag; }
};
```

`TaggedObjectStorage` provides hash-based storage with iterator access. Domain uses separate storages for nodes, elements, constraints, patterns.

### 2.4 MovableObject (Serialization)

All FEM classes inherit `MovableObject` for parallel/database support:

```cpp
class MovableObject {
    int classTag;   // Runtime type identifier
    int dbTag;      // Database identifier
public:
    virtual int sendSelf(int commitTag, Channel &channel) = 0;
    virtual int recvSelf(int commitTag, Channel &channel, FEM_ObjectBroker &broker) = 0;
};
```

`FEM_ObjectBroker` is the factory that reconstructs objects from serialized data using classTag.

### 2.5 Iterator Pattern

Domain provides abstract iterators (hide storage internals):

```cpp
NodeIter &nodeIter = domain.getNodes();
while ((Node *nd = nodeIter()) != 0) { ... }

ElementIter &eleIter = domain.getElements();
while ((Element *ele = eleIter()) != 0) { ... }
```

---

## 3. Domain Layer (Model Definition)

### 3.1 Domain (`domain/domain/Domain.h`)

Central container for the entire FEM model.

**Collections managed:**
- Nodes (TaggedObjectStorage)
- Elements (TaggedObjectStorage)
- SP_Constraints (single-point BCs)
- MP_Constraints (multi-point constraints)
- LoadPatterns (time-scaled load containers)
- Parameters (sensitivity parameters)
- Recorders

**Key state:**
```cpp
double currentTime;
double committedTime;
double dT;                              // time increment
int currentGeoTag;                      // tracks structural changes
bool hasDomainChangedFlag;
Vector *theEigenvalues;                 // from eigen analysis
int lastChannel;                        // parallel support
```

**Key methods:**
```cpp
// Model building
bool addNode(Node *);
bool addElement(Element *);
bool addSP_Constraint(SP_Constraint *);
bool addMP_Constraint(MP_Constraint *);
bool addLoadPattern(LoadPattern *);
void addRecorder(Recorder &);

// Removal
void clearAll();
Element *removeElement(int tag);
Node *removeNode(int tag);

// Access
Node *getNode(int tag);
Element *getElement(int tag);
NodeIter &getNodes();
ElementIter &getElements();
int getNumNodes();
int getNumElements();

// Analysis interface
void applyLoad(double pseudoTime);      // Scale all patterns by time
void setLoadConstant();                 // Lock current loads
int commit();                           // All nodes + elements commit
int revertToLastCommit();
int update();                           // All elements update from trial

// Eigenvalue analysis
int eigen(int numMode, bool generalized = true, bool findSmallest = true);
const Vector &getEigenvalues();
double getEigenvalue(int mode);

// Output
int record(bool = false);              // Trigger all recorders
void setCommitTag(int);

// Rayleigh damping
int setRayleighDampingFactors(double alphaM, double betaK, double betaK0, double betaKc);

// Graph
Graph &getNodeGraph();
Graph &getElementGraph();
bool hasDomainChanged();
int hasDomainChangeStamp;
```

### 3.2 Node (`domain/node/Node.h`)

DOF container at a spatial location.

**State vectors (trial + committed):**
```cpp
Vector *Crd;          // original coordinates (nDim)
Vector *commitDisp;   // committed displacement
Vector *commitVel;    // committed velocity
Vector *commitAccel;  // committed acceleration
Vector *trialDisp;    // trial displacement
Vector *trialVel;     // trial velocity
Vector *trialAccel;   // trial acceleration
Vector *unbalLoad;    // unbalanced (residual) force at node
Matrix *mass;         // nodal mass matrix
double alphaM;        // Rayleigh mass damping factor
Matrix **theEigenvectors;  // mode shapes
```

**Key methods:**
```cpp
int getNumberDOF();
const Vector &getCrds();

// Trial state
int setTrialDisp(const Vector &);
int setTrialVel(const Vector &);
int setTrialAccel(const Vector &);
int incrTrialDisp(const Vector &);      // Incremental update during Newton
int incrTrialVel(const Vector &);
int incrTrialAccel(const Vector &);

// Access
const Vector &getTrialDisp();
const Vector &getTrialVel();
const Vector &getTrialAccel();
const Vector &getDisp();                // committed
const Vector &getVel();                 // committed
const Vector &getAccel();               // committed
const Vector &getUnbalancedLoad();
const Vector &getUnbalancedLoadIncInertia();

// Mass and damping
int setMass(const Matrix &);
const Matrix &getMass();
int setRayleighDampingFactor(double alphaM);
const Matrix &getDamp();

// State management
int commitState();
int revertToLastCommit();
int revertToStart();

// Eigenvector
int setNumEigenvectors(int numVectorsToStore);
int setEigenvector(int mode, const Vector &eigenVector);
const Vector &getEigenvector(int mode);

// DOF Group (analysis link)
void setDOF_GroupPtr(DOF_Group *);
DOF_Group *getDOF_GroupPtr();
```

### 3.3 Constraints

**SP_Constraint** (single-point — prescribed DOF value):
```cpp
class SP_Constraint : public DomainComponent {
    int nodeTag;        // which node
    int dofNumber;      // which DOF (0-based)
    double valueR;      // reference value
    double valueC;      // current value = valueR * load_factor
    bool isConstant;    // time-invariant?

    double getValue();
    int applyConstraint(double loadFactor);
};
```

**MP_Constraint** (multi-point — ties DOFs between nodes):
```cpp
class MP_Constraint : public DomainComponent {
    int nodeRetained;
    int nodeConstrained;
    Matrix *constraint;     // [C] matrix: {u_c} = [C]{u_r}
    ID *constrDOF;          // constrained DOF indices
    ID *retainDOF;          // retained DOF indices
};
```

### 3.4 Load Patterns and Time Series

**LoadPattern** — container for loads scaled by a TimeSeries:
```cpp
class LoadPattern : public DomainComponent {
    TimeSeries *theSeries;                  // f(time) scaling
    TaggedObjectStorage *theNodalLoads;
    TaggedObjectStorage *theElementalLoads;
    TaggedObjectStorage *theSPs;

    void applyLoad(double pseudoTime) {
        double factor = theSeries->getFactor(pseudoTime);
        // apply each NodalLoad * factor
        // apply each ElementalLoad * factor
        // apply each SP_Constraint * factor
    }
};
```

**TimeSeries** hierarchy:
```cpp
class TimeSeries : public MovableObject, public TaggedObject {
    virtual double getFactor(double pseudoTime) = 0;
    virtual double getDuration() { return 0.0; }
};

// Concrete:
class ConstantSeries;      // returns constant factor
class LinearSeries;        // returns factor * pseudoTime
class PathTimeSeries;      // interpolated from data file
class TrigSeries;          // sin((2π/period)*(t-tStart) + shift)
```

**NodalLoad:**
```cpp
class NodalLoad : public Load {
    int myNode;       // target node tag
    Vector *load;     // force components
};
```

---

## 4. Element Layer

### 4.1 Element Base (`element/Element.h`)

Abstract base class. Every element must implement:

```cpp
class Element : public DomainComponent {
    // Connectivity
    virtual int getNumExternalNodes() = 0;
    virtual const ID &getExternalNodes() = 0;     // node tags
    virtual Node **getNodePtrs() = 0;             // node pointers
    virtual int getNumDOF() = 0;                  // total element DOF

    // Domain setup
    virtual void setDomain(Domain *theDomain);    // resolve node pointers

    // State management
    virtual int commitState() = 0;
    virtual int revertToLastCommit() = 0;
    virtual int revertToStart() = 0;
    virtual int update();                         // recompute from current node state

    // Stiffness and mass
    virtual const Matrix &getTangentStiff() = 0;  // current tangent K_T
    virtual const Matrix &getInitialStiff() = 0;  // initial K_0
    virtual const Matrix &getMass();               // mass matrix M
    virtual const Matrix &getDamp();               // damping matrix C (default: Rayleigh)

    // Forces
    virtual void zeroLoad() = 0;
    virtual int addLoad(ElementalLoad *theLoad, double loadFactor) = 0;
    virtual int addInertiaLoadToUnbalance(const Vector &accel) = 0;
    virtual const Vector &getResistingForce() = 0;           // F_int
    virtual const Vector &getResistingForceIncInertia() = 0;  // F_int + M*a + C*v

    // Response (for recorders)
    virtual Response *setResponse(const char **argv, int argc, OPS_Stream &output);
    virtual int getResponse(int responseID, Information &info);
};
```

### 4.2 Truss Element (`element/truss/Truss.h`)

Canonical simple element — axial-only bar.

```cpp
class Truss : public Element {
    // Connectivity
    int connectedExternalNodes[2];     // 2-node element
    Node *theNodes[2];                 // resolved pointers

    // Section/Material
    UniaxialMaterial *theMaterial;
    double A;                          // cross-sectional area
    double L;                          // current length
    double rho;                        // mass per unit length

    // Static workspace (avoids allocation in assembly loop)
    static Matrix trussM2, trussM4, trussM6, trussM12;
    static Vector trussV2, trussV4, trussV6, trussV12;

    // Key methods
    int update() {
        double strain = computeCurrentStrain();  // from node disps
        theMaterial->setTrialStrain(strain);
        return 0;
    }

    const Matrix &getTangentStiff() {
        double E = theMaterial->getTangent();
        double k = A * E / L;
        // Build K = k * n⊗n  (outer product of direction cosines)
        // Place in translational DOF blocks
    }

    const Vector &getResistingForce() {
        double stress = theMaterial->getStress();
        double force = A * stress;
        // F = force * n  (along element axis)
    }

    const Matrix &getMass() {
        if (cMass) {
            // Consistent: (rho*L/6)*[2I, I; I, 2I]
        } else {
            // Lumped: rho*L/2 on each diagonal translational DOF
        }
    }
};
```

### 4.3 Element Type Hierarchy

```
Element (abstract)
├── Truss, CorotTruss
├── elasticBeamColumn (Euler-Bernoulli, linear)
├── forceBeamColumn (fiber-section, force-based)
├── dispBeamColumn (fiber-section, displacement-based)
├── ShellMITC4, ShellDKGQ, ShellNLDKGQ
├── FourNodeQuad, EnhancedQuad, NineNodeMixedQuad
├── Brick, BbarBrick, TwentyNodeBrick
├── ZeroLength, ZeroLengthSection
├── Joint2D, Joint3D
├── TFP (triple friction pendulum)
├── ElastomericBearing
└── ... (hundreds of elements)
```

### 4.4 Coordinate Transformation (`coordTransformation/CrdTransf.h`)

Maps element-local to global frame. Used by beam/frame elements.

```cpp
class CrdTransf : public TaggedObject, public MovableObject {
    virtual int initialize(Node *node1, Node *node2) = 0;
    virtual int update() = 0;  // Recompute on deformation

    virtual double getInitialLength() = 0;
    virtual double getDeformedLength() = 0;

    // Local ↔ Global conversions
    virtual const Vector &getBasicTrialDisp() = 0;
    virtual const Vector &getBasicTrialVel() = 0;
    virtual const Vector &getBasicTrialAccel() = 0;
    virtual const Vector &getGlobalResistingForce(const Vector &basicForce, const Vector &p0) = 0;
    virtual const Matrix &getGlobalStiffMatrix(const Matrix &basicStiff, const Vector &basicForce) = 0;
    virtual const Matrix &getInitialGlobalStiffMatrix(const Matrix &basicStiff) = 0;
};

// Implementations:
class LinearCrdTransf2d;   // Small deformation (fixed local axes)
class LinearCrdTransf3d;
class CorotCrdTransf2d;    // Corotational (axes rotate with element)
class CorotCrdTransf3d;
class PDeltaCrdTransf2d;   // P-delta geometric nonlinearity
class PDeltaCrdTransf3d;
```

---

## 5. Material Layer

### 5.1 Material Hierarchy

```
Material (abstract base)
├── UniaxialMaterial      # 1D stress-strain
│   ├── ElasticMaterial   # σ = E·ε
│   ├── Steel01           # Bilinear with kinematic hardening (Menegotto-Pinto)
│   ├── Steel02           # Giuffré-Menegotto-Pinto with isotropic hardening
│   ├── Concrete01        # Kent-Park unconfined
│   ├── Concrete02        # Linear tension softening
│   ├── ElasticPP         # Elastic-perfectly-plastic
│   ├── Hardening         # Kinematic + isotropic hardening
│   ├── Hysteretic        # Trilinear backbone with pinching
│   ├── Parallel          # Materials in parallel (additive)
│   ├── Series            # Materials in series (reciprocal)
│   └── ... (100+ models)
│
├── NDMaterial            # Multi-dimensional (2D/3D stress-strain)
│   ├── ElasticIsotropic
│   ├── J2Plasticity
│   ├── DruckerPrager
│   ├── PressureIndependMultiYield
│   ├── PressureDependMultiYield
│   ├── PlaneStress / PlaneStrain adapters
│   └── ...
│
└── SectionForceDeformation   # Section-level (force-deformation)
    ├── ElasticSection2d/3d
    ├── FiberSection2d/3d     # Fiber discretization
    ├── GenericSection1d
    └── ...
```

### 5.2 UniaxialMaterial Interface (`material/uniaxial/UniaxialMaterial.h`)

```cpp
class UniaxialMaterial : public Material {
    virtual int setTrialStrain(double strain, double strainRate = 0.0) = 0;
    virtual int setTrialStrain(double strain, double temperature, double strainRate);

    virtual double getStrain() = 0;
    virtual double getStress() = 0;
    virtual double getTangent() = 0;          // dσ/dε
    virtual double getInitialTangent() = 0;

    virtual int commitState() = 0;
    virtual int revertToLastCommit() = 0;
    virtual int revertToStart() = 0;

    virtual UniaxialMaterial *getCopy() = 0;  // Deep clone
};
```

### 5.3 NDMaterial Interface (`material/nD/NDMaterial.h`)

```cpp
class NDMaterial : public Material {
    virtual int setTrialStrain(const Vector &strain) = 0;
    virtual int setTrialStrain(const Vector &strain, const Vector &rate);
    virtual int setTrialStrainIncr(const Vector &strain);

    virtual const Vector &getStrain() = 0;
    virtual const Vector &getStress() = 0;
    virtual const Matrix &getTangent() = 0;       // [C] = dσ/dε
    virtual const Matrix &getInitialTangent() = 0;

    virtual int commitState() = 0;
    virtual int revertToLastCommit() = 0;
    virtual int revertToStart() = 0;

    virtual NDMaterial *getCopy() = 0;
    virtual NDMaterial *getCopy(const char *type) = 0;  // "PlaneStress", "3D", etc.
    virtual const char *getType() = 0;
    virtual int getOrder() = 0;  // number of stress/strain components
};
```

### 5.4 SectionForceDeformation (`material/section/SectionForceDeformation.h`)

Section-level interface (force-deformation, not stress-strain):

```cpp
class SectionForceDeformation : public Material {
    virtual int setTrialSectionDeformation(const Vector &deforms) = 0;
    virtual const Vector &getSectionDeformation() = 0;
    virtual const Vector &getStressResultant() = 0;           // N, M_y, M_z, V_y, V_z, T
    virtual const Matrix &getSectionTangent() = 0;            // section flexibility/stiffness
    virtual const Matrix &getInitialTangent() = 0;

    virtual int commitState() = 0;
    virtual int revertToLastCommit() = 0;
    virtual int revertToStart() = 0;

    virtual SectionForceDeformation *getCopy() = 0;
    virtual const ID &getType() = 0;       // response types (SECTION_RESPONSE_P, _MZ, etc.)
    virtual int getOrder() = 0;            // number of response quantities
};
```

**FiberSection**: Discretizes cross-section into fibers, each with a UniaxialMaterial. Section stress resultants computed by integrating fiber stresses over area.

---

## 6. Linear Algebra Primitives (`matrix/`)

### 6.1 Vector (`matrix/Vector.h`)

```cpp
class Vector {
    double *theData;
    int sz;
public:
    Vector(int size);
    Vector(const Vector &);
    Vector(double *data, int size);  // wrap existing array (no copy)

    int Size() const;
    double &operator()(int x);       // 1-based access
    double &operator[](int x);       // 0-based access

    // Arithmetic
    Vector operator+(const Vector &);
    Vector operator-(const Vector &);
    Vector &operator+=(const Vector &);
    double operator^(const Vector &);  // dot product
    Vector &operator=(const Vector &);

    // Linear algebra
    double Norm() const;
    int Normalize();
    int addVector(double factThis, const Vector &other, double factOther);
    // theData = factThis * theData + factOther * other.theData

    int addMatrixVector(double factThis, const Matrix &m, const Vector &v, double factV);
    // theData = factThis * theData + factV * m * v

    int addMatrixTransposeVector(double factThis, const Matrix &m, const Vector &v, double factV);
    // theData = factThis * theData + factV * m^T * v

    // Assembly
    int Assemble(const Vector &V, const ID &rows, double fact = 1.0);
};
```

### 6.2 Matrix (`matrix/Matrix.h`)

```cpp
class Matrix {
    double *data;
    int numRows, numCols;
public:
    Matrix(int nRows, int nCols);
    Matrix(double *data, int nRows, int nCols);  // wrap existing

    double &operator()(int row, int col);  // 0-based

    // Arithmetic
    Matrix operator*(const Matrix &);     // matrix multiply
    Vector operator*(const Vector &);     // matrix-vector multiply
    Matrix operator+(const Matrix &);
    Matrix &operator+=(const Matrix &);

    // Assembly
    int Assemble(const Matrix &V, const ID &rows, const ID &cols, double fact = 1.0);

    // Solvers
    int Solve(const Vector &b, Vector &x);  // Solve A*x = b
    int Invert(Matrix &result);

    // Utilities
    void Zero();
    int addMatrix(double factThis, const Matrix &other, double factOther);
    int addMatrixProduct(double factThis, const Matrix &A, const Matrix &B, double factB);
    int addMatrixTripleProduct(double factThis, const Matrix &T, const Matrix &A, double factA);
    // result += factA * T^T * A * T  (used heavily for coordinate transformation)
};
```

### 6.3 ID (`matrix/ID.h`)

Integer array (used for DOF indices, node tags, etc.):

```cpp
class ID {
    int *data;
    int sz;
public:
    ID(int size);
    int &operator()(int x);   // 0-based
    int &operator[](int x);   // 0-based
    int Size() const;
    int getLocation(int value) const;  // find index of value
};
```

---

## 7. Analysis Framework

### 7.1 Complete Analysis Pipeline

```
User calls: analysis.analyze(numSteps)
│
├─ [ONCE] initialize()
│   ├─ ConstraintHandler::handle()
│   │   Creates FE_Element for each Element (wrappers for assembly)
│   │   Creates DOF_Group for each Node (groups free/fixed DOFs)
│   │
│   ├─ DOF_Numberer::numberDOF()
│   │   Builds Graph from FE_Element connectivity
│   │   Applies reordering (RCM, AMD, plain sequential)
│   │   Assigns global equation numbers to free DOFs
│   │
│   └─ LinearSOE::setSize(Graph)
│       Allocates K(nEqn × nEqn), b(nEqn), x(nEqn)
│       Storage format depends on solver (dense, banded, sparse)
│
└─ [PER STEP]
    ├─ Domain::applyLoad(pseudoTime)
    │   For each LoadPattern: factor = timeSeries.getFactor(t)
    │   Apply NodalLoads, ElementalLoads, SP_Constraints scaled by factor
    │
    ├─ Integrator::newStep(dt)
    │   Static: increment load factor
    │   Newmark: form effective stiffness K_eff, effective force F_eff
    │
    ├─ Algorithm::solveCurrentStep()    [ITERATION LOOP]
    │   ├─ Integrator::formTangent()
    │   │   SOE.zeroA()
    │   │   For each FE_Element:
    │   │     K_e = element.getTangentStiff()  (or getInitialStiff for modified Newton)
    │   │     SOE.addA(K_e, element_DOF_IDs)   (scatter into global K)
    │   │   For Newmark: also adds M, C terms to K_eff
    │   │
    │   ├─ Integrator::formUnbalance()
    │   │   SOE.zeroB()
    │   │   For each FE_Element:
    │   │     F_e = element.getResistingForce()
    │   │     SOE.addB(F_e, element_DOF_IDs)    (F_int contribution)
    │   │   For each DOF_Group:
    │   │     P = node.getUnbalancedLoad()
    │   │     SOE.addB(P, node_DOF_IDs)         (F_ext contribution)
    │   │   b = F_ext - F_int  (residual)
    │   │
    │   ├─ LinearSOE::solve()
    │   │   Solve K * ΔU = b for ΔU
    │   │
    │   ├─ AnalysisModel::setResponse(ΔU)
    │   │   For each DOF_Group:
    │   │     node.incrTrialDisp(ΔU_i)          (update trial)
    │   │
    │   ├─ AnalysisModel::updateDomain()
    │   │   Domain::update() → Element::update() for all elements
    │   │   (elements recompute strain/stress from current node disps)
    │   │
    │   └─ ConvergenceTest::test()
    │       Check ||R|| < tol  or  ||ΔU|| < tol  or  ΔU^T·R < tol
    │       Return: converged / continue / failed
    │
    ├─ Domain::commit()
    │   For all nodes: node.commitState()
    │   For all elements: element.commitState()
    │
    └─ Domain::record(commitTag, currentTime)
        For all recorders: recorder.record(...)
```

### 7.2 AnalysisModel (`analysis/model/AnalysisModel.h`)

Bridge between Domain (physical model) and equation system (mathematical model).

```cpp
class AnalysisModel {
    TaggedObjectStorage *theFEs;      // FE_Elements (one per element + constraint elements)
    TaggedObjectStorage *theDOFs;     // DOF_Groups (one per node)
    Graph *myDOFGraph;                // DOF connectivity graph
    Graph *myGroupGraph;              // DOF_Group connectivity graph

    bool addFE_Element(FE_Element *);
    bool addDOF_Group(DOF_Group *);

    int getNumEqn();                  // total free equations

    void setResponse(const Vector &disp, const Vector &vel, const Vector &accel);
    void incrResponse(const Vector &disp, const Vector &vel, const Vector &accel);
    int updateDomain();
    int commitDomain();
};
```

**FE_Element**: Wrapper around Element for assembly. Maps element stiffness/force into global equation numbers.

**DOF_Group**: Groups a node's DOFs. Maps local DOF index to global equation number. Tracks which DOFs are free vs constrained.

### 7.3 Algorithm (`analysis/algorithm/equiSolnAlgo/`)

```cpp
class EquiSolnAlgo : public SolutionAlgorithm {
    ConvergenceTest *theTest;
    virtual int solveCurrentStep() = 0;
    virtual int setConvergenceTest(ConvergenceTest *) = 0;
};
```

**Linear** — single solve (no iteration):
```cpp
int Linear::solveCurrentStep() {
    theIntegrator->formTangent();
    theIntegrator->formUnbalance();
    theSOE->solve();
    theModel->setResponse(theSOE->getX(), ...);
    theModel->updateDomain();
}
```

**Newton** — full Newton-Raphson:
```cpp
int Newton::solveCurrentStep() {
    theTest->start();
    do {
        theIntegrator->formTangent(tangentFlag);  // CURRENT_TANGENT or INITIAL_TANGENT
        theIntegrator->formUnbalance();
        theSOE->solve();
        theModel->incrResponse(theSOE->getX(), ...);
        theModel->updateDomain();
    } while (theTest->test() == -1);  // -1 = continue, 0 = converged, -2 = failed
}
```

**Other algorithms**: ModifiedNewton (reuse K), BFGS (quasi-Newton), Broyden, KrylovNewton, SecantNewton, PeriodicNewton.

### 7.4 Integrator Hierarchy

```
Integrator (abstract)
├── IncrementalIntegrator
│   ├── StaticIntegrator
│   │   ├── LoadControl          # Fixed load increment per step
│   │   ├── DisplacementControl  # Fixed displacement increment at a DOF
│   │   ├── ArcLength            # Arc-length (Riks) for snap-through
│   │   ├── MinUnbalDispNorm     # Minimum unbalanced displacement norm
│   │   └── ...
│   └── TransientIntegrator
│       ├── Newmark              # Newmark-β (average/linear acceleration)
│       ├── CentralDifference    # Explicit
│       ├── HHT                  # Hilber-Hughes-Taylor (numerical damping)
│       ├── GeneralizedAlpha     # Chung-Hulbert (controlled dissipation)
│       ├── TRBDF2               # TR-BDF2 (L-stable)
│       ├── Collocation          # Collocation method
│       └── ...
└── EigenIntegrator              # For eigenvalue analysis
```

**IncrementalIntegrator** key interface:
```cpp
class IncrementalIntegrator : public Integrator {
    virtual int newStep(double deltaT) = 0;   // Prepare for new step
    virtual int update(const Vector &deltaU);  // Update with solution increment
    virtual int commit();                       // Accept step

    // Form system matrices (assemble into LinearSOE)
    int formTangent(int statusFlag = CURRENT_TANGENT);
    int formUnbalance();

    // Individual element/node assembly
    virtual int formEleTangent(FE_Element *);
    virtual int formNodTangent(DOF_Group *);
    virtual int formEleResidual(FE_Element *);
    virtual int formNodUnbalance(DOF_Group *);
};
```

**Newmark integrator** key operations:
```cpp
// newStep(): Form effective stiffness
K_eff = K + (gamma/(beta*dt))*C + (1/(beta*dt*dt))*M

// formUnbalance(): Form effective force
F_eff = F_ext - F_int + M*(...) + C*(...)

// update(): After solving K_eff * ΔU = F_eff
U_{n+1} += ΔU
V_{n+1} = gamma/(beta*dt) * (U_{n+1} - U_n) + (1 - gamma/beta)*V_n + dt*(1 - gamma/(2*beta))*A_n
A_{n+1} = 1/(beta*dt*dt) * (U_{n+1} - U_n) - 1/(beta*dt)*V_n - (1/(2*beta) - 1)*A_n

// commit(): Save for next step
Ut = U_{n+1}; Vt = V_{n+1}; At = A_{n+1}
```

### 7.5 System of Equations (`system_of_eqn/`)

```cpp
class LinearSOE : public MovableObject {
    virtual int setSize(Graph &theGraph) = 0;   // Allocate based on connectivity
    virtual int addA(const Matrix &, const ID &, double fact = 1.0) = 0;  // Scatter K_e into K
    virtual int addB(const Vector &, const ID &, double fact = 1.0) = 0;  // Scatter f_e into f
    virtual int addM(const Matrix &, const ID &, double fact = 1.0);       // Mass scatter
    virtual int solve() = 0;                    // Solve K*x = b
    virtual int zeroA() = 0;
    virtual int zeroB() = 0;
    virtual const Vector &getX() = 0;           // Solution vector
    virtual const Vector &getB() = 0;           // RHS vector
    virtual int setB(const Vector &, double fact = 1.0) = 0;
    virtual double normRHS() = 0;               // ||b||
};
```

**Solver implementations:**
| Class | Storage | Method | Best For |
|-------|---------|--------|----------|
| FullGenLinSOE | Dense | LAPACK LU | Small systems, teaching |
| ProfileSPDLinSOE | Skyline/Profile | Cholesky | SPD, moderate size |
| BandGenLinSOE | Banded | LAPACK banded LU | Narrow bandwidth |
| BandSPDLinSOE | Banded SPD | LAPACK banded Cholesky | SPD, narrow band |
| SparseGenColLinSOE | Sparse CSC | SuperLU | General sparse |
| UmfpackGenLinSOE | Sparse | UMFPACK | General sparse, unsymmetric |
| MumpsSOE | Sparse | MUMPS | Parallel, large-scale |
| PetriSOE | Sparse | PETSc | Massively parallel |

### 7.6 Convergence Tests (`convergenceTest/`)

```cpp
class ConvergenceTest : public MovableObject {
    virtual int start() = 0;        // Initialize for new step
    virtual int test() = 0;         // Check; return 0=converged, -1=continue, -2=failed
    virtual int getNumTests() = 0;  // Iterations so far
    virtual int getMaxNumTests() = 0;
    virtual double getRatioNumToMax() = 0;
    virtual const Vector &getNorms() = 0;  // Norm history
};
```

| Test | Criterion | Formula |
|------|-----------|---------|
| CTestNormUnbalance | Residual norm | ‖R‖ < tol |
| CTestNormDispIncr | Displacement increment | ‖ΔU‖ < tol |
| CTestEnergyIncr | Energy increment | ΔU^T · R < tol |
| CTestRelativeNormUnbalance | Relative residual | ‖R‖/‖R_0‖ < tol |
| CTestRelativeNormDispIncr | Relative displacement | ‖ΔU‖/‖ΔU_0‖ < tol |
| CTestRelativeEnergyIncr | Relative energy | ΔU^T·R / ΔU_0^T·R_0 < tol |
| CTestFixedNumIter | Fixed iterations | Always iterate N times |

### 7.7 Constraint Handler (`handler/`)

```cpp
class ConstraintHandler : public MovableObject {
    virtual int handle(const ID *nodesLast = 0) = 0;
    virtual int update() = 0;
    virtual int applyLoad() = 0;
    virtual int doneNumberingDOF() = 0;
};
```

| Handler | Method | Pros | Cons |
|---------|--------|------|------|
| PlainHandler | Direct elimination | Simple, exact | Only homogeneous SP |
| PenaltyConstraintHandler | Large diagonal penalty | Any constraint | Ill-conditioning |
| LagrangeConstraintHandler | Lagrange multipliers | Exact, any constraint | Adds equations |
| TransformationConstraintHandler | Coordinate transform | Exact, efficient | Complex implementation |

### 7.8 DOF Numberer and Graph Numbering

```cpp
class DOF_Numberer : public MovableObject {
    GraphNumberer *theGraphNumberer;
    virtual int numberDOF(int lastDOF_Group = -1) = 0;
};

class GraphNumberer {
    virtual const ID &number(Graph &, int lastVertex = -1) = 0;
};

// Implementations:
class PlainNumberer;     // Sequential (no optimization)
class RCM;               // Reverse Cuthill-McKee (minimize bandwidth)
class AMD;               // Approximate Minimum Degree (minimize fill-in)
class SimpleNumberer;    // Simple sequential
```

---

## 8. Recorder System

### 8.1 Recorder Base (`recorder/Recorder.h`)

```cpp
class Recorder : public MovableObject, public TaggedObject {
    virtual int record(int commitTag, double timeStamp) = 0;
    virtual int restart() = 0;
    virtual int flush() = 0;
    virtual int domainChanged() { return 0; }
};
```

### 8.2 NodeRecorder (`recorder/NodeRecorder.h`)

```cpp
class NodeRecorder : public Recorder {
    ID *theDofs;               // DOF indices to record
    ID *theNodalTags;          // Node tags to record
    int dataFlag;              // What to record: disp(0), vel(1), accel(2), reaction(3), etc.
    OPS_Stream *theOutputHandler;
    double deltaT;             // Minimum time between recordings
    double relDeltaTTol;       // Relative tolerance for deltaT
    bool echoTimeFlag;         // Include time in first column
    TimeSeries **theTimeSeries; // Optional time series correction
    int numValidNodes;

    int record(int commitTag, double timeStamp) {
        // For each node in theNodalTags:
        //   Get response (disp/vel/accel/reaction based on dataFlag)
        //   Extract requested DOFs
        //   Write to theOutputHandler
    }
};
```

### 8.3 ElementRecorder

```cpp
class ElementRecorder : public Recorder {
    ID *eleID;                  // Element tags
    OPS_Stream *theOutputHandler;
    char **responseArgs;        // Response types (e.g., "force", "strain", "stress")
    int numArgs;
    Response **theResponses;    // Pre-built response objects (for efficiency)
    double deltaT;
    bool echoTimeFlag;
};
```

### 8.4 Output Handlers

```cpp
class OPS_Stream {
    virtual int write(const Vector &) = 0;
    virtual int write(const Matrix &) = 0;
    virtual int tag(const char *) = 0;
};

// Implementations:
class FileStream;       // Write to text file
class DataFileStream;   // Binary data file
class XmlFileStream;    // XML format
class DatabaseStream;   // SQL database
class StandardStream;   // stdout
class TCP_Stream;       // Network stream
```

---

## 9. Parallel Computing (`actor/`)

### 9.1 Channel

```cpp
class Channel {
    virtual int sendObj(int tag, MovableObject &, ChannelAddress * = 0) = 0;
    virtual int recvObj(int tag, MovableObject &, FEM_ObjectBroker &, ChannelAddress * = 0) = 0;
    virtual int sendVector(int tag, const Vector &, ChannelAddress * = 0) = 0;
    virtual int recvVector(int tag, Vector &, ChannelAddress * = 0) = 0;
    virtual int sendMatrix(int tag, const Matrix &, ChannelAddress * = 0) = 0;
    virtual int recvMatrix(int tag, Matrix &, ChannelAddress * = 0) = 0;
    virtual int sendID(int tag, const ID &, ChannelAddress * = 0) = 0;
    virtual int recvID(int tag, ID &, ChannelAddress * = 0) = 0;
};

// Implementations: TCP_Socket, MPI_Channel, UDP_Socket
```

### 9.2 Domain Decomposition

OpenSees supports **subdomain-based parallelism**:
- `Subdomain` extends `Domain` + `Element` (a subdomain is an element at the macro level)
- Each subdomain runs on a separate process
- Communication via condensed interface DOFs (static condensation at subdomain boundary)
- `PartitionedDomain` manages the global decomposed domain

---

## 10. Class Tag System (`classTags.h`)

Every concrete class gets a unique integer classTag (runtime type identification):

```cpp
#define ELE_TAG_Truss              1
#define ELE_TAG_ElasticBeam2d      3
#define MAT_TAG_ElasticMaterial    1
#define MAT_TAG_Steel01            500
#define INTEGRATOR_TAGS_Newmark    1
// ... hundreds of tags
```

Used by `FEM_ObjectBroker` to reconstruct objects from serialized data (factory pattern).

---

## 11. Key Architectural Insights for oneFEM

### 11.1 What to Copy Faithfully

1. **Trial/commit/revert interface** on Node, Element, Material — this is non-negotiable for Newton convergence and FE² nesting
2. **Strategy composition** for Analysis — keep Algorithm, Integrator, System, Numberer, Test, Handler as independent pluggable objects
3. **Element interface**: `getTangentStiff()`, `getResistingForce()`, `getInitialStiff()`, `getMass()`, `update()`, `commitState()`, `revertToLastCommit()`
4. **Material interface**: `setTrialStrain()`, `getStress()`, `getTangent()`, `commitState()`, `revertToLastCommit()`, `getCopy()`
5. **Section interface**: bridge between element and material for beam/frame elements
6. **CrdTransf**: essential for beam elements (local↔global, geometric nonlinearity)
7. **Assembly by scatter**: element stiffness → global K via DOF index mapping
8. **Separation of FE_Element/DOF_Group from Element/Node**: analysis-level wrappers that track equation numbers (though oneFEM can simplify this)

### 11.2 What to Simplify in Python

1. **No MovableObject/Channel** — Python has pickle/json for serialization; no MPI parallelism initially
2. **No FEM_ObjectBroker** — Python has dynamic typing; no need for class tag factory
3. **TaggedObjectStorage → dict** — Python dicts give O(1) by default
4. **ID → list or numpy array** — no need for custom integer array class
5. **Static workspace matrices** (e.g., `trussM6`) — Python/numpy handles allocation efficiently enough; premature optimization
6. **Iterator pattern** — Python's `for node in domain.nodes` is natural
7. **OPS_Stream** — Python has file I/O, pandas, etc.
8. **Graph/Numberer** — Start with plain numbering; add RCM via scipy later

### 11.3 Assembly Flow (the heart of FEM)

```
For each element e:
    1. K_e = element.getTangentStiff()   # (nDOF_e × nDOF_e) matrix
    2. dofs_e = element.getDOFs()         # global DOF indices for this element
    3. For each (i, j) in K_e:
         K_global[dofs_e[i], dofs_e[j]] += K_e[i, j]   # scatter

For each node n:
    4. F_global[dofs_n] += node.getUnbalancedLoad()     # external forces
    5. F_global[dofs_n] -= element.getResistingForce()   # internal forces (via element loop)

Solve: K_global[uu,uu] * ΔU = F_global[uu]  (uu = free DOFs)

For each node n:
    6. node.incrTrialDisp(ΔU[dofs_n])
For each element e:
    7. element.update()  # recompute strain/stress from new node disps
```

### 11.4 Newton-Raphson Flow (nonlinear)

```
# At each load step:
Integrator.newStep()          # Increment load factor or time

# Newton iteration loop:
while not converged:
    K = assemble tangent stiffness from all elements
    R = F_ext - F_int         # residual (unbalance)

    Solve: K * ΔU = R

    U_trial += ΔU             # increment trial displacement
    Element.update()           # elements recompute with new U_trial

    if ||R|| < tol:           # (or ||ΔU|| or ΔU^T·R)
        break

# Converged:
Domain.commit()               # trial → committed for all objects
```

### 11.5 Newmark Dynamic Flow

```
# Initialization (step 0):
M * a_0 = F_0 - C * v_0 - K * u_0    # initial acceleration

# Each time step (t → t+Δt):
1. Form K_eff = (1/(β·Δt²))·M + (γ/(β·Δt))·C + K
2. Form F_eff = F_{t+Δt} + M·[...] + C·[...]     # from committed state
3. Newton iterations on K_eff * U_{t+Δt} = F_eff
4. V_{t+Δt} = γ/(β·Δt)·(U_{t+Δt}-U_t) + (1-γ/β)·V_t + Δt·(1-γ/(2β))·A_t
5. A_{t+Δt} = 1/(β·Δt²)·(U_{t+Δt}-U_t) - 1/(β·Δt)·V_t - (1/(2β)-1)·A_t
6. Commit: U_t←U_{t+Δt}, V_t←V_{t+Δt}, A_t←A_{t+Δt}
```

---

## 12. Mapping: OpenSees C++ → oneFEM Python

| OpenSees (C++) | oneFEM (Python) | Notes |
|----------------|-----------------|-------|
| `Domain` | `Domain` (model/main.py) | Central container |
| `Node` | `Node` base + `Node{nDim}_{nDOF}` | Same specialization pattern |
| `Element` | `Element` (model/element/main.py) | Same interface |
| `UniaxialMaterial` | Material base + uniaxial/ | Same trial/commit |
| `NDMaterial` | material/nD/ | Same interface |
| `SectionForceDeformation` | Section (element/section/) | Same interface |
| `LoadPattern + TimeSeries` | pattern/ + tseries/ | Same pattern |
| `SP_Constraint` | model/constraint/ | Boundary conditions |
| `MP_Constraint` | (future) | Multi-point constraints |
| `StaticAnalysis` | `Analysis` (analysis/main.py) | Unified for now |
| `EquiSolnAlgo` | algorithm/ (Linear, Newton) | Same strategy |
| `StaticIntegrator` | integrator/static/ | LoadControl, etc. |
| `TransientIntegrator` | integrator/dynamic/ | Newmark, etc. |
| `LinearSOE` | system/ (FullGeneral, UmfPack) | Same solver abstraction |
| `ConvergenceTest` | test/ (UnbalancedLoad, DispIncr) | Same interface |
| `ConstraintHandler` | constraints/ (Plain) | Simplified |
| `DOF_Numberer` | numberer/ (Plain) | Simplified |
| `CrdTransf` | (future) | Needed for beams |
| `Recorder` | output/recorder/ | Same pattern |
| `Vector` | `_systools/data/vector.py` | Wraps numpy |
| `Matrix` | `_systools/data/matrix.py` | Wraps numpy |
| `ID` | Python list / numpy int array | No custom class needed |
| `TaggedObjectStorage` | Python dict | O(1) by default |
| `AnalysisModel` | (integrated into Analysis) | Simplified |
| `FE_Element` | (not needed) | Python simplification |
| `DOF_Group` | (not needed) | Python simplification |
| `MovableObject` | (not needed) | Use pickle if needed |
| `FEM_ObjectBroker` | (not needed) | Python dynamic typing |
| `Eigen` | analysis/eigen/ | scipy eigh/eigsh |
