from ANNarchy import *
from ANNarchy.extensions.bold.PredefinedModels import balloon_two_inputs
from ANNarchy.extensions.bold import BoldModel
from parameters import params, rng

setup(dt=params['dt'])
setup(num_threads=params['num_threads'])
setup(seed=params['seed'])
Constant('RS_v_r',params['RS_v_r'])
Constant('FS_v_r',params['FS_v_r'])



####################################################################################################################################
##################################################   NEURON MODELS   ###############################################################
####################################################################################################################################

Izhikevich2007RS = Neuron(
    parameters="""
        C        = 'RS_C'      : population
        k        = 'RS_k'      : population
        v_r      = 'RS_v_r'    : population
        v_t      = 'RS_v_t'    : population
        a        = 'RS_a'      : population
        b        = 'RS_b'      : population
        c        = 'RS_c'      : population
        d        = 'RS_d'      : population
        v_peak   = 'RS_v_peak' : population
        tau_ampa = 'tau_ampa'  : population
        tau_gaba = 'tau_gaba'  : population
        E_ampa   = 'E_ampa'    : population
        E_gaba   = 'E_gaba'    : population
        
        tau_syn  = 'tau_syn'   : population
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa : init = 0
        dg_gaba/dt = -g_gaba/tau_gaba : init = 0
        I_ampa     = -g_ampa*(v - E_ampa)
        I_gaba     = -g_gaba*(v - E_gaba)
        I          = g_exc + I_ampa + I_gaba
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I : init = RS_v_r
        du/dt      = a*(b*(v - v_r) - u) : init = 0
        
        tau_syn*dsyn/dt = -syn
        var_f  = pos(I_ampa) - 1.5*neg(I_gaba)
        var_r  = pos(I_ampa) 
        var_ra = pos(I_ampa) 
        var_rb = pos(I_ampa)**(1/3.)
    """,
    spike = "v >= v_peak",
    reset = """
        v = c
        u = u + d
    """,
    name = "Izhikevich2007RS",
    description = "RS cortical neuron model from Izhikevich (2007) with additional conductance based synapses.",
    extra_values=params
)

Izhikevich2007FS = Neuron(
    parameters="""
        C        = 'FS_C'      : population
        k        = 'FS_k'      : population
        v_r      = 'FS_v_r'    : population
        v_t      = 'FS_v_t'    : population
        v_b      = 'FS_v_b'    : population
        a        = 'FS_a'      : population
        b        = 'FS_b'      : population
        c        = 'FS_c'      : population
        d        = 'FS_d'      : population
        v_peak   = 'FS_v_peak' : population
        tau_ampa = 'tau_ampa'  : population
        tau_gaba = 'tau_gaba'  : population
        E_ampa   = 'E_ampa'    : population
        E_gaba   = 'E_gaba'    : population
        
        tau_syn  = 'tau_syn'   : population
    """,
    equations="""
        dg_ampa/dt = -g_ampa/tau_ampa : init = 0
        dg_gaba/dt = -g_gaba/tau_gaba : init = 0
        I_ampa     = -g_ampa*(v - E_ampa)
        I_gaba     = -g_gaba*(v - E_gaba)
        I          = g_exc + I_ampa + I_gaba
        C * dv/dt  = k*(v - v_r)*(v - v_t) - u + I : init = FS_v_r
        U_v        = if v<v_b: 0 else: b*(v - v_b)**3
        du/dt      = a*(U_v - u) : init = 0
        
        tau_syn*dsyn/dt = -syn
        var_f  = pos(I_ampa) - 1.5*neg(I_gaba)
        var_r  = pos(I_ampa)
        var_ra = r
        var_rb  = pos(I_ampa)**(1/3.)
    """,
    spike = "v >= v_peak",
    reset = """
        v = c
        u = u + d
    """,
    name = "Izhikevich2007FS",
    description = "FS cortical interneuron model from Izhikevich (2007) with additional conductance based synapses.",
    extra_values=params
)

InputNeuron = Neuron(
    parameters="""
        tau         = 'input_tau' : population
        offsetVal   = 0
        increaseVal = 0
    """,
    equations="""
        r=increaseVal*t/tau+offsetVal
    """,
    name = "InputNeuron",
    description = "Rate of neuron increases linearly, proportionally to offset.",
    extra_values=params
)

InputPoissonNeuron = Neuron(
    parameters="""
        tau         = 'input_tau' : population
        offsetVal   = 0
        increaseVal = 0
    """,
    equations = """
        p = Uniform(0.0, 1.0) * 1000.0 / dt
        rate=increaseVal*t/tau+offsetVal
    """,
    spike = """
        p <= rate
        """,
    reset = """
       p=0.0
    """,
    name = "InputPoissonNeuron",
    description = "Spiking rate of Poisson neuron increases linearly, proportionally to offset.",
    extra_values=params
)

### BOLD model only for single input recording
BoldModel_r = BoldModel(
parameters = """
""",
equations = """
    I_CBF         = sum(I_CBF)    : init=0
    I_CMRO2       = sum(I_CMRO2)  : init=0
""",
inputs=['I_CBF','I_CMRO2'],
output="I_CBF",
)
