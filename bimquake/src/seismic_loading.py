import numpy as np

def _soil_factors(soil_category):
    """ Calculate soil factors based on soil category.

        Parameters
        ----------
        soil_category : str
            Soil category ('A', 'B', 'C', 'D', 'E').

        Returns
        -------
        p1 : float
            Soil factor p1 based on soil category. 
            
        p2 : float
            Soil factor p2 based on soil category.
            
        Tc_func : function
            Function to calculate characteristic period Tc based on soil category. """
    
    if soil_category == 'A':
        return 1.0, 1.0, lambda Tc: Tc

    elif soil_category == 'B':
        return 1.4, 1.1, lambda Tc: Tc**(-0.2) * 1.1 * Tc

    elif soil_category == 'C':
        return 1.7, 1.05, lambda Tc: Tc**(-0.33) * 1.05 * Tc

    elif soil_category == 'D':
        return 2.4, 1.25, lambda Tc: Tc**(-0.5) * 1.25 * Tc

    elif soil_category == 'E':
        return 2.0, 1.15, lambda Tc: Tc**(-0.4) * 1.15 * Tc

def _get_Vr_and_Tr(service_life, class_str):
    """ Calculate design ground acceleration and return period based on service life and importance class.

        Parameters
        ----------
        service_life : float
            Service life of the building in years.

        class_str : str
            Importance class of the building ('I', 'II', 'III', 'IV').

        Returns
        -------
        Vr : float
            Design ground acceleration.

        TrSLV : float
            Return period corresponding to the service life. """
    
    print ("class:", class_str)
    match class_str:
        case 'I':
            cu = 0.7
        case 'II':
            cu = 1
        case 'III':
            cu = 1.5
        case 'IV':
            cu = 2

    Vr = cu * service_life
    TrSLV = - Vr / np.log (1 - 0.1)
    return Vr, TrSLV


def _iterate_return_periods_hazard_params(ParaTR, Tr, Trmax=2475, table=None):
    """ Interpolate hazard parameters for a specific return period.
    
        Parameters
        ----------
        ParaTR : np.ndarray
            Array of hazard parameters for predefined return periods.
            
        Tr : float
            Desired return period.
            
        Trmax : float, optional
            Maximum return period. Default is 2475.
            
        table : list of float, optional
            List of predefined return periods. Default is [30, 50, 72, 101, 140, 201, 475, 975, Trmax].
            
        Returns
        -------
        Parameters : np.ndarray
            Interpolated hazard parameters for the specified return period. """
    if table is None:
        table = [30, 50, 72, 101, 140, 201, 475, 975, Trmax]

    if Tr >= 3 and Tr < 30:
        lower = table[0]
        higher = table[1]
        idx = 0
        Parameters = (ParaTR[idx+1,:]-ParaTR[idx,:])/(np.log(higher/lower))*np.log(Tr/lower)+ParaTR[idx,:]
        return Parameters
    
    elif Tr >= 975:
        lower = table[-2]
        higher = table[-1]
        idx = len(table)-2
        Parameters = (ParaTR[idx+1,:]-ParaTR[idx,:])/(np.log(higher/lower))*np.log(Tr/lower)+ParaTR[idx,:]
        return Parameters
    
    else:
        for i in range(len(table)-1):
            if Tr >= table[i] and Tr < table[i+1]:
                lower = table[i]
                higher = table[i+1]
                idx = i
                Parameters = (ParaTR[idx+1,:]-ParaTR[idx,:])/(np.log(higher/lower))*np.log(Tr/lower)+ParaTR[idx,:]
                return Parameters
            

def _get_Sgeo_and_Cc(soli_category, Parameters):
    """ Calculate seismic site coefficient and soil correction factor based on soil category and parameters.
    
        Parameters
        ----------
        soli_category : str
            Soil category ('A', 'B', 'C', 'D', 'E').
            
        Parameters : np.ndarray
            Array of hazard parameters.
            
        Returns
        -------
        S_Sgeo : float  
            Seismic site coefficient.
            
        Cc : float
            Soil correction factor."""
    
    match soli_category:
        
        case 'A':
            S_Sgeo=1
            Cc=1
            
        case 'B':
            S_Sgeo=1.4-0.4*Parameters[1]*Parameters[0]

            if S_Sgeo<1:
                S_Sgeo=1
            elif S_Sgeo>1.2:
                S_Sgeo=1.2

            Cc=1.1*(Parameters[2]**(-0.2))

        case 'C':
            S_Sgeo=1.7-0.6*Parameters[1]*Parameters[0]

            if S_Sgeo<1:
                S_Sgeo=1
            elif S_Sgeo>1.5:
                S_Sgeo=1.5

            Cc=1.05*(Parameters[2]**(-0.33))

        case 'D':
            S_Sgeo=2.4-1.5*Parameters[1]*Parameters[0]

            if S_Sgeo<0.9:
                S_Sgeo=0.9
            elif S_Sgeo>1.8:
                S_Sgeo=1.8

            Cc=1.25*(Parameters[2]**(-0.5))

        case 'E':
            S_Sgeo=2-1.1*Parameters[1]*Parameters[0]
            if S_Sgeo<1:
                S_Sgeo=1
            elif S_Sgeo>1.6:
                S_Sgeo=1.6

            Cc=1.15*(Parameters[2]**(-0.4))
    return S_Sgeo, Cc

def _get_S_t(topographic_category):
    """ Calculate topographic amplification factor based on topographic category.
    
        Parameters
        ----------
        topographic_category : str
            Topographic category ('T1', 'T2', 'T3', 'T4').
            
        Returns
        -------
        S_t : float
            Topographic amplification factor. """
    
    match topographic_category:
        case 'T1':
            S_t=1
        case 'T2':
            S_t=1.2
        case 'T3':
            S_t=1.2
        case 'T4':
            S_t=1.4
    return S_t

def _get_ADRS(TC, Parameters, S_geo):
    """ Calculate Acceleration-Displacement Response Spectrum (ADRS) based on characteristic period and parameters.
    
        Parameters
        ----------
        TC : float
            Characteristic period TC.
            
        Parameters : np.ndarray
            Array of hazard parameters.
            
        S_geo : float
            Seismic site coefficient.
            
        Returns
        -------
        ADRS : np.ndarray
            Array representing the ADRS values.
            
        tstep : np.ndarray
            Array of time steps used in ADRS calculation. """
    
    TB=TC/3
    TD=4*Parameters[0]+1.6

    tstep = _get_tstep(TC, TB, TD)
    ntstep = len(tstep)

    ADRS = np.zeros((ntstep, 2))
    for i in range(ntstep):
        Ty = tstep[i]

        if Ty<TC/3:
            multiplier = Ty/(TB)+1/Parameters[1]*(1-Ty/(TB))
        elif Ty>=TC/3 and Ty<TC:
            multiplier = 1
        elif Ty>=TC and Ty<TD:
            multiplier = TC/Ty
        elif Ty>=TD:
            multiplier = TC*TD/(Ty**2)
            
        ADRS[i,1]=Parameters[0]*S_geo*Parameters[1]*multiplier
        ADRS[i,0]=(Ty**2)*ADRS[i,1]*9.81/(4*(np.pi**2))*1000
    return ADRS, tstep



def _get_tstep(TC, TB, TD):
    """ Generate time steps for ADRS calculation based on characteristic periods.
    
        Parameters
        ----------
        TC : float
            Characteristic period TC.
            
        TB : float
            Characteristic period TB.
            
        TD : float
            Characteristic period TD.
            
        Returns
        -------
        tstep : np.ndarray
            Array of time steps for ADRS calculation. """
    
    tstep0 = np.arange(0, TB, TB/20)
    tstep1 = np.arange(TB, TC, 2/60*TC)
    tstep2 = np.arange(TC, TD, (TD-TC)/20)
    tstep3 = np.arange(TD, 4, (4-TD)/20)
    tstep = np.concatenate((tstep0, tstep1, tstep2, tstep3, np.array([4])))
    return tstep

def iterate_return_periods_ADRS(
    Mass, F_eq, design_params, delta_eq, kel,
    soil_category
    ):

    """ Iterate over return periods to compute the Acceleration-Displacement Response Spectrum (ADRS) and related parameters.
    
        Parameters
        ----------
        Masse : float
            Mass of the structure.
            
        F_eq : np.ndarray
            Equivalent force array.
            
        design_params : dict
            Dictionary containing design parameters including 'tstep', 'ParaTR', and 'S_geo'.
            
        delta_eq : np.ndarray
            Equivalent displacement array.
            
        kel : float
            Elastic stiffness of the structure.
            
        soil_category : str
            Soil category ('A', 'B', 'C', 'D', 'E').
            
        Returns
        -------
        Tr : float
            Return period corresponding to the best match.
            
        ag : float
            Design ground acceleration for the best match.
            
        ADRS_TR : np.ndarray
            Acceleration-Displacement Response Spectrum (ADRS) for the best match.
            
        Sda : np.ndarray
            Displacement response spectrum for the best match.
            
        Saa : np.ndarray
            Acceleration response spectrum for the best match. """
    
    tstep = design_params["tstep"]
    ParaTR = design_params["ParaTR"]
    ParaTR = ParaTR.values[:, 1:]
    S_geo = design_params["S_geo"]

    Trmax = 2475
    years = np.arange(3, Trmax+10000, 1)

    M = Mass
    Tx = 2*np.pi*np.sqrt(M / (kel * 9.81))

    # soil functions
    _, p1, Tc_func = _soil_factors(soil_category)

    dxstar_t = np.zeros(len(years))

    # -----------------------------
    # 1. find matching return period
    # -----------------------------
    for i, Tr in enumerate(years):

        Params = _iterate_return_periods_hazard_params(ParaTR, Tr)
        ag, p1_h, Tc0 = Params

        TC = Tc_func(Tc0)
        TD = ag/9.81 * 4 + 1.6

        Sae, Sde = _get_spectral_points(Tx, ag, p1_h, TC, TD, S_geo)

        S_eq = F_eq / (9.81 * M)
        axstar = S_eq[1]
        dxstar_y = delta_eq[1]

        if delta_eq[1] > Sde and F_eq[1] < Sae * M * 9.81:
            dxstar_t[i] = Sde
        else:
            q = Sae / axstar

            if Tx < TC:
                dxstar_t[i] = Sde / q * (1 + (q - 1) * TC / Tx)
            else:
                dxstar_t[i] = Sde

    # best match
    pos = np.argmin(np.abs(dxstar_t - delta_eq[2]))
    Tr = years[pos]

    # -----------------------------
    # 2. recompute final spectrum
    # -----------------------------
    ag, p1, Tc0 = _iterate_return_periods_hazard_params(ParaTR, Tr)

    TC = Tc_func(Tc0)
    TD = ag/9.81 * 4 + 1.6

    N = len(tstep)
    ADRS_TR = np.zeros((N, 2))

    for i, T in enumerate(tstep):
        S, D = _get_spectral_points(T, ag, p1, TC, TD, S_geo)
        # D =  T**2 * S * 9.81 / (4 * np.pi**2) * 1e3
        ADRS_TR[i] = [D, S]

    # -----------------------------
    # final evaluation at structure period
    # -----------------------------
    S_eq = F_eq / (9.81 * M)
    axstar = S_eq[1]
    dxstar_y = delta_eq[1]
    Sae, Sde = _get_spectral_points(Tx, ag, p1, TC, TD, S_geo)
    # Se = T**2 * Sae * 9.81 / (4 * np.pi**2) * 1e3

    if delta_eq[2] > Sde and F_eq[2] > Sae * M * 9.81:
        dxstar = Sde
        Saa = ADRS_TR[:, 1]
        Sda = ADRS_TR[:, 0]
    else:
        q = Sae / axstar

        if Tx < TC:
            dxstar = Sde / q * (1 + (q - 1) * TC / Tx)
        else:
            dxstar = Sde

        mu = dxstar / dxstar_y

        # R_mu = np.ones(len(tstep))
        # mask = tstep < TC
        # R_mu[mask] = (mu - 1) * tstep[mask] / TC + 1

        Nt = len(tstep)
        R_mu = np.zeros(Nt)

        for j in range(Nt):
            if tstep[j]<TC:
                R_mu[j]=(mu-1)*tstep[j]/TC+1
            else:
                R_mu[j]=mu


        Saa = ADRS_TR[:, 1] / R_mu
        Sda = mu * (ADRS_TR[:, 0] / R_mu)

    return Tr, ag, ADRS_TR, Sda, Saa


def _get_spectral_points(T, ag, p1, TC, TD, S_geo):
        """ Calculate spectral acceleration and displacement points based on period and parameters.
        
            Parameters
            ----------
            T : float
                Period for which to calculate spectral points.
                
            ag : float
                Design ground acceleration.
                
            p1 : float
                Soil factor p1 based on soil category.
                
            TC : float
                Characteristic period TC.
                
            TD : float
                Characteristic period TD.
                
            S_geo : float
                Seismic site coefficient.
                
            Returns
            -------
            Sae : float
                Spectral acceleration at period T.
                
            Sde : float
                Spectral displacement at period T. """
        
        if T < TC / 3:
            Sae = ag * S_geo * p1 * (T/(TC/3) + 1/p1 * (1 - T/(TC/3)))
        elif T < TC:
            Sae = ag * S_geo * p1
        elif T < TD:
            Sae = ag * S_geo * p1 * TC / T
        else:
            Sae = ag * S_geo * p1 * TC * TD / T**2
        Sde=T**2*Sae*9.81/(4*np.pi**2)*10**3

        return Sae, Sde


def get_seismic_loading(
                      ParaTR,
                      service_life,
                      importance_class,
                      soil_category,
                      topographic_category,
                      total_height):

    """ Calculate seismic loading parameters based on design parameters and site characteristics.
    
        Parameters
        ----------
        ParaTR : np.ndarray
            Array of hazard parameters for predefined return periods.
            
        service_life : float
            Service life of the building in years.
            
        importance_class : str
            Importance class of the building ('I', 'II', 'III', 'IV').
            
        soil_category : str
            Soil category ('A', 'B', 'C', 'D', 'E').
            
        topographic_category : str
            Topographic category ('T1', 'T2', 'T3', 'T4').
            
        total_height : float
            Total height of the building in meters.
            
        Returns
        -------
        Se_SLV_T : np.ndarray
            Spectral acceleration at service life and period T.
            
        ag_SLV : float
            Design ground acceleration for service life.
            
        S_geo : float
            Seismic site coefficient.
            
        ADRS : np.ndarray
            Acceleration-Displacement Response Spectrum (ADRS) for service life.
            
        tstep : np.ndarray
            Array of time steps used in ADRS calculation.
            
        spectral_point_func : function
            Function to calculate spectral points based on period T.
            
        TC : float
            Characteristic period TC based on soil category and parameters. """
    
    print("servicee life:", service_life)
    print("importance class:", importance_class)
    _, TrSLV = _get_Vr_and_Tr(service_life, importance_class)
    ParaTR_ = ParaTR.values[:, 1:]
    params = _iterate_return_periods_hazard_params(ParaTR_, TrSLV)
    ag_SLV, p_1, p_2 = params
    S_Sgeo, Cc = _get_Sgeo_and_Cc(soil_category, params)
    S_t = _get_S_t(topographic_category)

    S_geo = S_t * S_Sgeo

    TC  = p_2 * Cc  # needed for pushover
    ADRS, tstep = _get_ADRS(TC, params, S_geo) # needed for pushover
    
    # TC_SLV = (p_2**(-0.2)) * 1.1 * p_2   #If I get it well, this is for fdr one soil category (B), TC is more general
    TD = ag_SLV / 9.81 * 4 + 1.6
    H = total_height   #instead of hTOT      
    T = (H**(3/4))*0.05
    Se_SLV_T, _ = _get_spectral_points(T, ag_SLV, p_1, TC, TD, S_geo)
    spectral_point_func = lambda T: _get_spectral_points(T, ag_SLV, p_1, TC, TD, S_geo) # Here it was TC_SLV before

    return Se_SLV_T, ag_SLV, S_geo, ADRS, tstep, spectral_point_func, TC