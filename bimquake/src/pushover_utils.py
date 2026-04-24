import numpy as np
import math
from copy import deepcopy
import pandas as pd
import plot_utils


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
    
    match class_str:
        case 'I':
            cu=0.7
        case 'II':
            cu=1
        case 'III':
            cu=1.5
        case 'IV':
            cu=2

    Vr=cu*service_life
    TrSLV=-Vr/np.log(1-0.1)
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
            

def _get_Sgeo_and_Cc(soli_category, Parametri):
    """ Calculate seismic site coefficient and soil correction factor based on soil category and parameters.
    
        Parameters
        ----------
        soli_category : str
            Soil category ('A', 'B', 'C', 'D', 'E').
            
        Parametri : np.ndarray
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
            S_Sgeo=1.4-0.4*Parametri[1]*Parametri[0]

            if S_Sgeo<1:
                S_Sgeo=1
            elif S_Sgeo>1.2:
                S_Sgeo=1.2

            Cc=1.1*(Parametri[2]**(-0.2))

        case 'C':
            S_Sgeo=1.7-0.6*Parametri[1]*Parametri[0]

            if S_Sgeo<1:
                S_Sgeo=1
            elif S_Sgeo>1.5:
                S_Sgeo=1.5

            Cc=1.05*(Parametri[2]**(-0.33))

        case 'D':
            S_Sgeo=2.4-1.5*Parametri[1]*Parametri[0]

            if S_Sgeo<0.9:
                S_Sgeo=0.9
            elif S_Sgeo>1.8:
                S_Sgeo=1.8

            Cc=1.25*(Parametri[2]**(-0.5))

        case 'E':
            S_Sgeo=2-1.1*Parametri[1]*Parametri[0]
            if S_Sgeo<1:
                S_Sgeo=1
            elif S_Sgeo>1.6:
                S_Sgeo=1.6

            Cc=1.15*(Parametri[2]**(-0.4))
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

def _compute_global_vulnerability(D,X,S,V,G,deltau,N,NZ,Masses,alt_s,red_F,ADRS,inc,deltared,n_floors,alpha,passo,Parameters,ParaTR,tstep,S_geo,TC,terreno,algorithm):
    """ Compute various structural properties and parameters for the building model.
    
        Parameters
        ----------
        D : list of np.ndarray
            List of wall dimensions for each floor.
            
        X : list of np.ndarray
            List of wall center coordinates for each floor.
            
        S : list of np.ndarray
            List of wall strengths for each floor.
            
        V : list of np.ndarray
            List of wall volumes for each floor.
            
        G : list of np.ndarray
            List of wall material properties for each floor.
            
        deltau : list of np.ndarray
            List of wall maximum allowable displacements for each floor.
            
        N : int
            Number of floors.
            
        NZ : list of int
            List of number of walls per floor.
            
        Masses : np.ndarray
            Array of masses for each floor.
            
        alt_s : list of float
            List of story heights.
            
        red_F : float
            Force reduction factor.
            
        ADRS : np.ndarray
            Array of acceleration-displacement response spectrum values.
            
        inc : float
            Time increment for analysis.
            
        deltared : float
            Reduced damping value.
            
        n_floors : int
            Number of floors for analysis.
            
        alpha : list of np.ndarray
            List of wall orientations for each floor.
            
        passo : float
            Time step for analysis.
            
        Parameters : np.ndarray
            Array of seismic parameters.
            
        ParaTR : np.ndarray
            Array of seismic hazard parameters.
            
        tstep : float
            Time step for analysis.
            
        S_geo : float
            Geotechnical site coefficient.
            
        TC : float
            Characteristic period for the site.
            
        terreno : str
            Soil type.
            
        algorithm : str
            Analysis algorithm to use.
            
        Returns
        -------
        Various computed structural properties and parameters. """
    
    MTOT=np.sum(Masses)
    
    # Initialize main floor(storey) properties (center of mass, center of rigidity) 
    TX=np.zeros(N) 
    TY=np.zeros(N) 
    KXel=np.zeros(N) 
    KYel=np.zeros(N) 
    XP=np.zeros(N) 
    YP=np.zeros(N) 
    EX=np.zeros(N) 
    EY=np.zeros(N) 
    JX=np.zeros(N)
    JY=np.zeros(N)
    PM=np.zeros(N) 
    vr_el_x=np.zeros(N) 
    Hel_x=np.zeros(N) 
    vr_el_y=np.zeros(N) 
    Hel_y=np.zeros(N) 

    # Computation of floor properties
    deltax_el = []
    Fx_el = [] 
    Hx_el = [] 
    RX = []

    deltay_el = []
    Fy_el = []
    Hy_el = []
    RY = []
    K = []

    for j in range(N):
        X0=X[j] 
        D0=D[j] 
        S0=S[j] 
        G0=G[j] 
        deltau0=deltau[j]
        alpha0=alpha[j]
        Gvar1=G[j]
        V0=V[j]
        SF=0
        SS=0
        SX=0
        SY=0
        for i in range(NZ[j]):
            SF=S0[i]*D0[i,0]*D0[i,1]
            SS=SS+SF
            SX=SX+SF*X0[i,0]
            SY=SY+SF*X0[i,1]
        TX[j]=SX/SS 
        TY[j]=SY/SS

        #Stifness of the walls
        K0=np.zeros((NZ[j],2))
        for i in range(NZ[j]):
            Gvar0 = Gvar1[i,2]/Gvar1[i,0]
            theta=alpha0[i]*np.pi/180
            K0[i,0]=Gvar1[i,0]*D0[i,0]*D0[i,1]/(1.2*V0[i]*(1+1/(1.2*Gvar0)*(V0[i]/D0[i,0])**2))*np.cos(theta)**2
            K0[i,1]=Gvar1[i,0]*D0[i,0]*D0[i,1]/(1.2*V0[i]*(1+1/(1.2*Gvar0)*(V0[i]/D0[i,0])**2))*np.sin(theta)**2 

        K.append(K0)
        KXel[j]=np.sum(K0[:,0])
        KYel[j]=np.sum(K0[:,1])
        

        XP[j]=np.sum(K0[:,1]*X0[:,0])/KYel[j]
        YP[j]=np.sum(K0[:,0]*X0[:,1])/KXel[j]
        EX[j]=TX[j]-XP[j]
        EY[j]=TY[j]-YP[j]
        
        X1=np.sum(K0[:,0]*X0[:,1]**2)
        Y1=np.sum(K0[:,1]*X0[:,0]**2)
        JX[j]=X1-KXel[j]*YP[j]**2
        JY[j]=Y1-KYel[j]*XP[j]**2
        IY=JY
        IX=JX
        PM[j]=JX[j]+JY[j]
        
        
        H0=np.zeros(NZ[j])
        
        for k in range(2):
            R0=np.zeros((NZ[j],2))
            for i in range(NZ[j]):
                if k==0:
                    R0[i,0]=1+EY[j]*KXel[j]*(X0[i,1]-YP[j])/PM[j]
                    R0[i,1]=EY[j]*KXel[j]*(X0[i,1]-YP[j])/PM[j]
                elif k==1:
                    R0[i,0]=EX[j]*KYel[j]*(X0[i,0]-XP[j])/PM[j]
                    R0[i,1]=1+EX[j]*KYel[j]*(X0[i,0]-XP[j])/PM[j]

            DM=0
            delta0=np.zeros(NZ[j])
            for i in range(NZ[j]):
                if k==0:
                    if alpha0[i] < 45 or alpha0[i] >= 135:
                        H0[i]=D0[i,0]*D0[i,1]*G0[i,1]*(1+S0[i]/(1.5*G0[i,1]))**0.5 
                    else:
                        H0[i]=0
                elif k==1:
                    if alpha0[i] < 45 or alpha0[i] >= 135:
                        H0[i]=0
                    else:
                        H0[i]=D0[i,0]*D0[i,1]*G0[i,1]*(1+S0[i]/(1.5*G0[i,1]))**0.5
                        
                delta0[i]=H0[i]/K0[i,k]
                if delta0[i]>deltau0[i,0]:
                    delta0[i]=deltau0[i,0]
                    H0[i]=deltau0[i,0]*K0[i,k]

                DM1=delta0[i]/R0[i,k]
                if not math.isnan(DM1):
                    if DM1<DM or DM == 0:
                        DM=DM1

            if k==0:
                deltax=delta0
                vr_el_x[j]=DM
                Fx0_i=vr_el_x[j]*R0[:,k]*K0[:,k]
                H0x = deepcopy(H0)
                deltax_el.append(deltax)
                Fx_el.append(Fx0_i)
                Hx_el.append(H0x)
                Hel_x[j]=KXel[j]*vr_el_x[j]
                RX.append(R0)
                

            elif k==1:
                deltay=delta0
                vr_el_y[j]=DM
                Fy0_i=vr_el_y[j]*R0[:,k]*K0[:,k]
                H0y = deepcopy(H0)
                deltay_el.append(deltay)
                Fy_el.append(Fy0_i)
                Hy_el.append(H0y)
                Hel_y[j]=KYel[j]*vr_el_y[j]
                RY.append(R0)
                
            

    # Main characteristics for pushover analysis:
    step=10000
    alpha_start=0.01

    for j in range(2):
        # Initiate stiffness, forces, displacements 
        KE=np.zeros((N,step)) # stiffness
        HE=np.zeros((N,step)) # horizontal force
        vr=np.zeros((N,step)) # displacement
        NC=0 # Number of collapsed walls
        delta2=0
        if j==0:
            Fx=np.zeros((N,step))
            vrx=np.zeros((N,step))
            match algorithm:
                case 'incr':
                    Fx[:,0]=alpha_start*MTOT*red_F
                case 'add':
                    Fx[:,0]=passo/9.81*red_F
            KE[:,0]=KXel
            for i in range(N):
                HE[i,0]=np.sum(Fx[:i+1,0])
            vrx[:,0]=HE[:,0]/KE[:,0]
            F=Fx
            vr=vrx
            LX=np.zeros((step,3))
            LX_el=np.zeros((step,3))
            R1=RX

        elif j==1:
            Fy=np.zeros((N,step))
            vry=np.zeros((N,step))
            match algorithm:
                case 'incr':
                    Fy[:,0]=alpha_start*MTOT*red_F
                case 'add':
                    Fy[:,0]=passo/9.81*red_F

            KE[:,0]=KYel
            for i in range(N):
                HE[i,0]=np.sum(Fy[:i+1,0])
            vry[:,0]=HE[:,0]/KE[:,0]
            F=Fy
            vr=vry
            LY=np.zeros((step,3))
            LY_el=np.zeros((step,3))
            R1=RY
        LoopCounter=0
        Hmax=-1
        piani_coll=0

        while HE[N-1,LoopCounter]>=Hmax*(1-deltared) and piani_coll == 0:
            match algorithm:
                case 'incr':
                    F[:,LoopCounter+1]=F[:,LoopCounter]*inc
                case 'add':
                    F[:,LoopCounter+1]=F[:,LoopCounter]+passo/9.81*red_F
            for i in range(N):
                vr[i,LoopCounter+1]=np.sum(F[:i+1,LoopCounter+1])/KE[i,LoopCounter]

            C=np.zeros((np.max(NZ),N))
            E=np.zeros((np.max(NZ),N))
            delta2=np.zeros((np.max(NZ),N))

            for k in range(N):
                D0=D[k]
                deltau0=deltau[k]
                R0=R1[k]
                X0=X[k]
                H0=np.concatenate((Hx_el[k].reshape(-1, 1), Hy_el[k].reshape(-1, 1)), axis=1)
                alpha0=alpha[k]
                K0=K[k]
                delta0_el=np.zeros(NZ[k])

                for i in range(NZ[k]):
                    if alpha0[i] < 45 or alpha0[i] >= 135:

                        deltael=deltax_el[k]               
                    else:
                        deltael=deltay_el[k]
                    delta0_el[i]=deltael[i]

                for i in range(NZ[k]):

                    I=True
                    if NC!=0:
                        if j==0:

                            if i in LX[:NC,0]:
                                I = False

                        elif j==1:
                            if i in LY[:NC,0]:
                                I = False
                    if I:
                        if k==0:
                            if alpha0[i] < 45 or alpha0[i] >= 135:

                                theta=alpha0[i]*np.pi/180
                                delta2[i,k]=np.abs(vr[k,LoopCounter+1]*R0[i,0]*np.cos(theta))           
                            else:
                                theta=(alpha0[i]-90)*np.pi/180
                                delta2[i,k]=np.abs(vr[k,LoopCounter+1]*R0[i,1]*np.cos(theta))
                        else:
                            if alpha0[i] < 45 or alpha0[i] >= 135:

                                theta=alpha0[i]*np.pi/180
                                delta2[i,k]=np.abs(vr[k,LoopCounter+1]*R0[i,0]*np.cos(theta))
                            else:
                                theta=(alpha0[i]-90)*np.pi/180
                                delta2[i,k]=np.abs(vr[k,LoopCounter+1]*R0[i,1]*np.cos(theta))

                        if delta2[i,k]>delta0_el[i] and delta2[i,k]<deltau0[i,j]:
                            if j==0:
                                if alpha0[i] < 45 or alpha0[i] >= 135:

                                    C[i,k]=H0[i,j]
                                else:
                                    C[i,k]=H0[i,1]
                            elif j==1:
                                if alpha0[i] < 45 or alpha0[i] >= 135:

                                    C[i,k]=H0[i,0]
                                else:
                                    C[i,k]=H0[i,j]

                            E[i,k]=C[i,k]/delta2[i,k]

                        elif delta2[i,k]<delta0_el[i] and delta2[i,k]<deltau0[i,j]:
                            if j==0:
                                if alpha0[i] < 45 or alpha0[i] >= 135:

                                    theta=alpha0[i]*np.pi/180

                                    C[i,k]=delta2[i,k]*K0[i,j]/(np.abs(np.cos(theta)))          
                                else:
                                    theta=(alpha0[i]-90)*np.pi/180

                                    C[i,k]=delta2[i,k]*K0[i,1]/(np.abs(np.cos(theta)))
                            elif j==1:
                                if alpha0[i] < 45 or alpha0[i] >= 135:

                                    theta=alpha0[i]*np.pi/180

                                    C[i,k]=delta2[i,k]*K0[i,0]/(np.abs(np.cos(theta)))
                                else:
                                    theta=(alpha0[i]-90)*np.pi/180

                                    C[i,k]=delta2[i,k]*K0[i,j]/(np.abs(np.cos(theta)))                                
                            E[i,k] = K0[i,j]

                        elif delta2[i,k]>deltau0[i,j]:
                            C[i,k]=0
                            E[i,k]=0
                            if j==0:
                                if NC == 0:
                                    NC = NC+1
                                    LX[NC-1,0]=i
                                    LX[NC-1,1]=-k+N-1
                                    LX[NC-1,2]=LoopCounter
                                else:
                                    array = np.abs(i-LX[:NC,0])+np.abs(-k+N-LX[:NC,1])
                                    a = np.min(array)
                                    pos = np.argmin(array)
                                    if a!=0 or (a==0 and LX[pos,1]!=-k+N):
                                        NC = NC+1
                                        LX[NC-1,0]=i
                                        LX[NC-1,1]=-k+N-1
                                        LX[NC-1,2]=LoopCounter

                            elif j==1:
                                if NC == 0:
                                    NC = NC+1
                                    LY[NC-1,0]=i
                                    LY[NC-1,1]=-k+N-1
                                    LY[NC-1,2]=LoopCounter
                                else:
                                    array = np.abs(i-LY[:NC,0])+np.abs(-k+N-LY[:NC,1])
                                    a = np.min(array)
                                    pos = np.argmin(array)
                                    if a!=0 or (a==0 and LY[pos,1]!=-k+N):
                                        NC = NC+1
                                        LY[NC-1,0]=i
                                        LY[NC-1,1]=-k+N-1
                                        LY[NC-1,2]=LoopCounter
                        if j==0:
                            if alpha0[i] < 45 or alpha0[i] >= 135:

                                HE[k,LoopCounter+1]=HE[k,LoopCounter+1]+C[i,k]          
                                KE[k,LoopCounter+1]=KE[k,LoopCounter+1]+E[i,k]
                        elif j==1:
                            if alpha0[i] >= 45 and alpha0[i] < 135:

                                HE[k,LoopCounter+1]=HE[k,LoopCounter+1]+C[i,k]
                                KE[k,LoopCounter+1]=KE[k,LoopCounter+1]+E[i,k]
                        if HE[N-1,LoopCounter+1]>Hmax:
                            Hmax=HE[N-1,LoopCounter+1]
                        
                vr[k,LoopCounter+1]=HE[k,LoopCounter+1]/KE[k,LoopCounter+1]
                if vr[k,LoopCounter+1]<vr[k,LoopCounter]:
                    vr[k,LoopCounter+1]=vr[k,LoopCounter]

                if vr[k,LoopCounter+1]>vr[k,LoopCounter] and HE[k,LoopCounter+1]<HE[k,LoopCounter]:
                    vr[k,LoopCounter+1]=vr[k,LoopCounter]

                if k==N-1:
                    if np.sum(vr[:k+1,LoopCounter+1])>np.sum(vr[:k+1,LoopCounter]) and HE[k,LoopCounter+1]<HE[k,LoopCounter]:
                        vr[:,LoopCounter+1]=vr[:,LoopCounter]
                    
                if j==0:
                    YP1=np.sum(E[:NZ[k],k]*X0[:,1])/KE[k,LoopCounter+1]
                    X1_1=np.sum(E[:NZ[k],k]*X0[:,1]**2)
                    EY1=TY[k]-YP1
                    JX1=X1_1-KE[k,LoopCounter+1]*YP1**2
                    PM1=JX1+JY[k]
                    R0[:,j]=1+EY1*KE[k,LoopCounter+1]*(X0[:,1]-YP1)/PM1
                    R0[:,1]=EY1*KE[k,LoopCounter+1]*(X0[:,1]-YP1)/PM1
                    R1[k]=R0
                    Fx[:,LoopCounter+1]=F[:,LoopCounter+1]
                    vrx[:,LoopCounter+1]=vr[:,LoopCounter+1]

                elif j==1:
                    XP1=np.sum(E[:NZ[k],k]*X0[:,0])/KE[k,LoopCounter+1]
                    Y1_1=np.sum(E[:NZ[k],k]*X0[:,0]**2)
                    EX1=TX[k]-XP1
                    JY1=Y1_1-KE[k,LoopCounter+1]*XP1**2
                    PM2=JX[k]+JY1
                    R0[:,j]=1+EX1*KE[k,LoopCounter+1]*(X0[:,0]-XP1)/PM2
                    R0[:,0]=EX1*KE[k,LoopCounter+1]*(X0[:,0]-XP1)/PM2
                    R1[k]=R0
                    Fy[:,LoopCounter+1]=F[:,LoopCounter+1]
                    vry[:,LoopCounter+1]=vr[:,LoopCounter+1]

            for i in range(N):
                if HE[i,LoopCounter+1]==0:
                    piani_coll=piani_coll+1
            LoopCounter=LoopCounter+1

            if piani_coll!=0:
                LoopCounter=LoopCounter-1
                if j==0:
                    L0=LX
                elif j==1:
                    L0=LY
                
                nonzero_elements = len(np.where(L0>0)[0])
                L0_1=np.zeros((nonzero_elements,3))
                for i in range(nonzero_elements):
                    if L0[i,2]!=LoopCounter:
                        L0_1[i,:]=L0[i,:]

                if j==0:
                    LX=L0_1
                else:
                    LY=L0_1
            if j==0:
                Kult_x=KE[:,:LoopCounter+1]
                Hult_x=HE[:,:LoopCounter+1]
                vr_ult_x=vr[:,:LoopCounter+1]
            elif j==1:
                Kult_y=KE[:,:LoopCounter+1]
                Hult_y=HE[:,:LoopCounter+1]
                vr_ult_y=vr[:,:LoopCounter+1]

    LX0 = LX[np.where(LX[:, 2]>0)[0], 0:2]
    LY0 = LY[np.where(LY[:, 2]>0)[0], 0:2]

    LX_reshaped = []
    for i in range(N):
        floor = LX0[np.where(LX0[:, 1]==i)[0], 0]
        if floor is None:
            floor = []
        LX_reshaped.append(floor)
    
    LX_reshaped = list(reversed(LX_reshaped))

    LY_reshaped = []
    for i in range(N):
        floor = LY0[np.where(LY0[:, 1]==i)[0], 0]
        if floor is None:
            floor = []
        LY_reshaped.append(floor)

    LY_reshaped = list(reversed(LY_reshaped))
    L = [LX_reshaped, LY_reshaped]

    Hult_x_TOT = np.concatenate((np.array([0]), Hult_x[N-1, :])) * 9.81
    Hult_y_TOT = np.concatenate((np.array([0]), Hult_y[N-1, :])) * 9.81

    Hult_TOT = [Hult_x_TOT, Hult_y_TOT]

    vr_ult_x_TOT = np.concatenate((np.array([0]), np.sum(vr_ult_x, axis=0))) * 1000
    vr_ult_y_TOT = np.concatenate((np.array([0]), np.sum(vr_ult_y, axis=0))) * 1000

    vr_ult_TOT = [vr_ult_x_TOT, vr_ult_y_TOT]

    k_ult_x_TOT=np.sum(Kult_x, axis=0)
    k_ult_y_TOT=np.sum(Kult_y, axis=0)

    k_ult_TOT = [k_ult_x_TOT, k_ult_y_TOT]
    return k_ult_TOT, Hult_TOT, vr_ult_TOT, L


def _calculate_bilinears(X, Y):
    """ Calculate bilinear curve approximations for pushover curves in X and Y directions.

        Parameters
        ----------
        X : list of np.ndarray
            List of x-coordinates for X and Y directions.

        Y : list of np.ndarray
            List of y-coordinates for X and Y directions.

        Returns
        -------
        x_coordinates : list of list
            List of x-coordinates for bilinear curves in X and Y directions.

        y_coordinates : list of list
            List of y-coordinates for bilinear curves in X and Y directions. """
    
    x_coordinates = []
    y_coordinates = []
    for i in range(2):
        x = X[i]
        y = Y[i]

        curve_area = _get_area_under_curve(x, y)
        slope = _get_slope(x, y)
        triangle_area = _get_triangle_area(slope, x)
        diff_area = triangle_area - curve_area
        w, h = _get_triangle_sides(slope, diff_area)
        x_coord, y_coord = _get_bilinear_coordinates(x, y, h, w)
        x_coordinates.append(x_coord)
        y_coordinates.append(y_coord)
        
    return x_coordinates, y_coordinates

def _get_area_under_curve(x, y):
    """ Calculate the area under a curve defined by points (x, y) using trapezoidal rule.
    
        Parameters
        ----------
        x : list or np.ndarray
            x-coordinates of the curve points.
            
        y : list or np.ndarray
            y-coordinates of the curve points.
            
        Returns
        -------
        area : float
            Area under the curve. """
    
    area = 0
    for i in range(len(x)-1):
        if x[i+1] == x[i]:
            continue
        lower_point = min(y[i], y[i+1])
        higher_point = max(y[i], y[i+1])
        width = x[i+1] - x[i]
        rectangle = width*lower_point
        triangle = width*(higher_point-lower_point)/2
        delta_area = rectangle+triangle
        area += delta_area
    return area


def _get_slope(x, y):
    """ Calculate the slope of a line defined by two points.
    
        Parameters
        ----------
        x : list or np.ndarray
            x-coordinates of the two points.
            
        y : list or np.ndarray
            y-coordinates of the two points.
            
        Returns
        -------
        slope : float
            Slope of the line. """
    
    slope = y[1] / x[1]
    return slope

def _get_triangle_area(slope, x_curve):
    """ Calculate the area of a triangle given its slope and base length.
    
        Parameters
        ----------
        slope : float
            Slope of the triangle.
            
        x_curve : list or np.ndarray
            x-coordinates defining the base of the triangle.
            
        Returns
        -------
        area : float
            Area of the triangle. """
    
    x_side = x_curve[-1] - x_curve[0]
    area = 1/2 * (x_side**2) * slope
    return area

def _get_triangle_sides(slope, area):
    """ Calculate the base and height of a triangle given its slope and area.
    
        Parameters
        ----------
        slope : float
            Slope of the triangle.
            
        area : float
            Area of the triangle.
            
        Returns
        -------
        w : float
            Base length of the triangle.
            
        h : float
            Height of the triangle. """
    
    h = np.sqrt(area*2*slope)
    w = h / slope
    return w, h

def _get_bilinear_coordinates(x_curve, y_curve, triangle_height, triangle_width):
    """ Calculate the coordinates of a bilinear curve approximation.
    
        Parameters
        ----------
        x_curve : list or np.ndarray
            x-coordinates of the original curve.
            
        y_curve : list or np.ndarray
            y-coordinates of the original curve.
            
        triangle_height : float
            Height of the triangle used for approximation.

        triangle_width : float
            Width of the triangle used for approximation.

        Returns
        -------
        x_coordinates : list
            x-coordinates of the bilinear curve.

        y_coordinates : list
            y-coordinates of the bilinear curve. """

    break_x = x_curve[-1] - triangle_width
    break_y = x_curve[-1]/x_curve[1]*y_curve[1] - triangle_height

    x_coordinates = [x_curve[0], break_x, np.max(x_curve)]
    y_coordinates = [y_curve[0], break_y, break_y]
    return x_coordinates, y_coordinates


def _get_data_for_pushover_results(ADRS, Parameters, Masses, vr_ult_TOT, Hult_TOT, delta_ult_eq, Hult_eq, kult_TOT, tstep, S_geo, TC, ParaTR, soil_category):
    """ Helper function to compute data for plotting pushover curves and response spectra.

        Parameters
        ----------
        ADRS : np.ndarray
            Array of acceleration-displacement response spectrum values.
            
        Parameters : list of float
            List of seismic parameters.
            
        Masses : np.ndarray
            Array of masses for each floor.
            
        vr_ult_TOT : list of np.ndarray
            List of ultimate displacements for X and Y directions.
            
        Hult_TOT : list of np.ndarray
            List of ultimate base shear forces for X and Y directions.
            
        delta_ult_eq : list of np.ndarray
            List of ultimate equivalent displacements for X and Y directions.
            
        Hult_eq : list of np.ndarray
            List of ultimate equivalent base shear forces for X and Y directions.
            
        kult_TOT : list of np.ndarray
            List of ultimate stiffness values for X and Y directions.
            
        tstep : np.ndarray
            Array of time steps.
        
        S_geo : float
            Geotechnical site factor.
            
        TC : float
            Characteristic period.
            
        ParaTR : pandas.DataFrame
            Parameters for the TR method.
            
        soil_category : str
            Soil category for the analysis.
            
        Returns
        -------
        Saa : list of np.ndarray
            List of spectral acceleration values for X and Y directions.

        Sda : list of np.ndarray
            List of spectral displacement values for X and Y directions.

        delta_ult_eq : list of np.ndarray
            List of ultimate equivalent displacements for X and Y directions.

        S_eq : list of np.ndarray
            List of equivalent spectral acceleration values for X and Y directions.

        dxstars : list of float
            List of adjusted displacement values for X and Y directions.

        Tr : list of np.ndarray
            List of TR response spectra for X and Y directions.

        IR : list of float
            List of safety indicies for X and Y directions.

        ADRS_TR : list of np.ndarray
            List of TR acceleration-displacement response spectra for X and Y directions.

        Sda_TR : list of np.ndarray
            List of TR spectral displacement values for X and Y directions.

        Saa_TR : list of np.ndarray
            List of TR spectral acceleration values for X and Y directions.        

        ag_TR : list of float
            List of peak ground accelerations for X and Y directions. """
    
    e_point = []
    k_el_TOT = []
    Saa = []
    Sda = []
    S_eq = []
    dxstars = []
    for i in range(2):
        diff=np.zeros(len(kult_TOT[i]))
        for j in range(len(kult_TOT[i])-1):
            diff[j]=kult_TOT[i][j+1]-kult_TOT[i][j]
        e_point.append(len(kult_TOT[i])-len(np.where(diff!=0)[0]))

        k_el_TOT.append((Hult_TOT[i][e_point[i]]/9.81)/(vr_ult_TOT[i][e_point[i]]/1000))

        TD=Parameters[i]/9.81*4+1.6

        TC_x=TC
        MTOT=np.sum(Masses)
        Tx=2*np.pi*(MTOT/(k_el_TOT[i]*9.81))**0.5
    
        S_eq_x=Hult_eq[i]/(9.81*MTOT)
        S_eq.append(S_eq_x)
        axstar=S_eq_x[1]
        dxstar_y=delta_ult_eq[i][1]

        Nt=len(tstep)
        R_mu=np.zeros(Nt)

        if Tx<TC_x/3:
            Sae_x=Parameters[0]*S_geo*Parameters[1]*(Tx/(TC_x/3)+1/Parameters[1]*(1-Tx/(TC_x/3)))
            Sde_x=Tx**2*Sae_x*9.81/(4*np.pi**2)*10**3
        elif Tx>TC_x/3 and Tx<TC_x:
            Sae_x=Parameters[0]*S_geo*Parameters[1]
            Sde_x=Tx**2*Sae_x*9.81/(4*np.pi**2)*10**3
        elif Tx>TC_x and Tx<TD:
            Sae_x=Parameters[0]*S_geo*Parameters[1]*TC_x/Tx
            Sde_x=Tx**2*Sae_x*9.81/(4*np.pi**2)*10**3
        elif Tx>TD:
            Sae_x=Parameters[0]*S_geo*Parameters[1]*TC_x*TD/Tx**2
            Sde_x=Tx**2*Sae_x*9.81/(4*np.pi**2)*10**3

        if (delta_ult_eq[i][2]-Sde_x)>0 and Hult_eq[i][2]>Sae_x*MTOT*9.81:
            dxstar_t=Sde_x
            qstar_x=1
            Saa_x=ADRS[:,1]
            Sda_x=ADRS[:,0]
        else:
            qstar_x=Sae_x/axstar
            if Tx<TC_x:
                dxstar_t=Sde_x/qstar_x*(1+(qstar_x-1)*TC_x/Tx)
            elif Tx>TC_x:
                dxstar_t=Sde_x
            mu_x=dxstar_t/dxstar_y
            
            for j in range(Nt):
                if tstep[j]<TC_x:
                    R_mu[j]=(mu_x-1)*tstep[j]/TC_x+1
                else:
                    R_mu[j]=mu_x
                    
            Saa_x=ADRS[:,1]/R_mu
            Sda_x=mu_x*(ADRS[:,0]/R_mu)
        Saa.append(Saa_x)
        Sda.append(Sda_x)
        dxstars.append(dxstar_t)

    Tr_x, ag_x_TR, ADRS_x_TR, Sda_x_TR, Saa_x_TR = _iterate_return_periods_ADRS(MTOT,Hult_eq[0],S_geo,delta_ult_eq[0],k_el_TOT[0],tstep,ParaTR,soil_category)
    Tr_y, ag_y_TR, ADRS_y_TR, Sda_y_TR, Saa_y_TR = _iterate_return_periods_ADRS(MTOT,Hult_eq[1],S_geo,delta_ult_eq[1],k_el_TOT[1],tstep,ParaTR,soil_category)
    Tr = [Tr_x, Tr_y]
    ADRS_TR = [ADRS_x_TR, ADRS_y_TR]
    Sda_TR = [Sda_x_TR, Sda_y_TR]
    Saa_TR = [Saa_x_TR, Saa_y_TR]

    IR_x = round(ag_x_TR/Parameters[0]*100)/100
    IR_y = round(ag_y_TR/Parameters[0]*100)/100

    IR = [IR_x, IR_y]
    ag_TR = [ag_x_TR, ag_y_TR]

    return Saa, Sda, delta_ult_eq, S_eq, dxstars, Tr, IR, ADRS_TR, Sda_TR, Saa_TR, ag_TR


def _iterate_return_periods_ADRS(Masse,F_x_eq,S_geo,delta_x_eq,kel_x,tstep,ParaTR,soil_category):
    """ Helper function to perform TR method iteration for seismic response spectrum.
    
        Parameters
        ----------
        Masse : float
            Total mass of the structure.
            
        F_x_eq : np.ndarray
            Equivalent base shear forces.
            
        S_geo : float
            Geotechnical site factor.
            
        delta_x_eq : np.ndarray
            Equivalent displacements.
            
        kel_x : float
            Elastic stiffness.
            
        tstep : np.ndarray
            Array of time steps.
            
        ParaTR : pandas.DataFrame
            Parameters for the TR method.
            
        soil_category : str
            Soil category for the analysis.
            
        Returns
        -------
        Tr : float
            Return period.
            
        ag_TR : float
            Peak ground acceleration.
            
        ADRS_TR : np.ndarray
            Acceleration-displacement response spectrum.
            
        Sda_x_TR : np.ndarray
            Spectral displacement values.
            
        Saa_x_TR : np.ndarray
            Spectral acceleration values. """
    
    Trmax=2475
    years = np.arange(3, Trmax+10000, 1)
    n=len(years)
    M=Masse
    dxstar_t=np.zeros(n)
    Sda_x=np.zeros(n)
    Nt=len(tstep)

    for i in range(n):

        Tr=years[i]
        Parametri0 = _iterate_return_periods_hazard_params(ParaTR, Tr)

        if Parametri0 is None:
            print(i)
        
        TD=Parametri0[0]/9.81*4+1.6

        match soil_category:
            case 'A':
                TC_x=Parametri0[2]
            case 'B':
                TC_x=Parametri0[2]**(-0.2)*1.1*Parametri0[2]
            case 'C':
                TC_x=Parametri0[2]**(-0.33)*1.05*Parametri0[2]
            case 'D':
                TC_x=Parametri0[2]**(-0.5)*1.25*Parametri0[2]
            case 'E':
                TC_x=Parametri0[2]**(-0.4)*1.15*Parametri0[2]

        Tx=2*np.pi*(M/(kel_x*9.81))**0.5
        S_x_eq=F_x_eq/(9.81*Masse)
        axstar=S_x_eq[1]
        dxstar_y=delta_x_eq[1]
        
        if Tx<TC_x/3:
            Sae_x=Parametri0[0]*S_geo*Parametri0[1]*(Tx/(TC_x/3)+1/Parametri0[1]*(1-Tx/(TC_x/3)))
            Sde_x=Tx**2*Sae_x*9.81/(4*np.pi**2)*10**3
        elif Tx>TC_x/3 and Tx<TC_x:
            Sae_x=Parametri0[0]*S_geo*Parametri0[1]
            Sde_x=Tx**2*Sae_x*9.81/(4*np.pi**2)*10**3
        elif Tx>TC_x and Tx<TD:
            Sae_x=Parametri0[0]*S_geo*Parametri0[1]*TC_x/Tx
            Sde_x=Tx**2*Sae_x*9.81/(4*np.pi**2)*10**3
        elif Tx>TD:
            Sae_x=Parametri0[0]*S_geo*Parametri0[1]*TC_x*TD/Tx**2
            Sde_x=Tx**2*Sae_x*9.81/(4*np.pi**2)*10**3
        
        if (delta_x_eq[1]-Sde_x)>0 and F_x_eq[1]<Sae_x*Masse*9.81:
            dxstar_t[i]=Sde_x
            qstar_x=1
        else:
            qstar_x=Sae_x/axstar
            if Tx<TC_x:
                dxstar_t[i]=Sde_x/qstar_x*(1+(qstar_x-1)*TC_x/Tx)
            elif Tx>TC_x:
                dxstar_t[i]=Sde_x
            mu_x=dxstar_t[i]/dxstar_y
            if Tx<TC_x:
                R_mu_x=(mu_x-1)*Tx/TC_x+1
            else:
                R_mu_x=mu_x
        
    diff = np.min(np.abs(dxstar_t-delta_x_eq[2]))
    pos = np.argmin(np.abs(dxstar_t-delta_x_eq[2]))
    Tr_x=pos+3
    Tr=Tr_x

    Parametri0 = _iterate_return_periods_hazard_params(ParaTR, Tr)
    TD=Parametri0[0]/9.81*4+1.6

    match soil_category:
        case 'A':
            TC=Parametri0[2]
        case 'B':
            TC=Parametri0[2]**(-0.2)*1.1*Parametri0[2]
        case 'C':
            TC=Parametri0[2]**(-0.33)*1.05*Parametri0[2]
        case 'D':
            TC=Parametri0[2]**(-0.5)*1.25*Parametri0[2]
        case 'E':
            TC=Parametri0[2]**(-0.4)*1.15*Parametri0[2]

    N=len(tstep)
    ADRS_TR=np.zeros((N,2))
    for i in range(N):
        T=tstep[i]    


        if T<TC/3:
            ADRS_TR[i,1]=Parametri0[0]*S_geo*Parametri0[1]*(T/(TC/3)+1/Parametri0[1]*(1-T/(TC/3)))
            ADRS_TR[i,0]=T**2*ADRS_TR[i,1]*9.81/(4*np.pi**2)*10**3
        elif T>=TC/3 and T<TC:
            ADRS_TR[i,1]=Parametri0[0]*S_geo*Parametri0[1]
            ADRS_TR[i,0]=T**2*ADRS_TR[i,1]*9.81/(4*np.pi**2)*10**3
        elif T>=TC and T<TD:
            ADRS_TR[i,1]=Parametri0[0]*S_geo*Parametri0[1]*TC/T
            ADRS_TR[i,0]=T**2*ADRS_TR[i,1]*9.81/(4*np.pi**2)*10**3
        elif T>=TD:
            ADRS_TR[i,1]=Parametri0[0]*S_geo*Parametri0[1]*TC*TD/T**2
            ADRS_TR[i,0]=T**2*ADRS_TR[i,1]*9.81/(4*np.pi**2)*10**3

    ag_TR=Parametri0[0]

    match soil_category:
        case 'A':
            TC_x=Parametri0[2]
        case 'B':
            TC_x=Parametri0[2]**(-0.2)*1.1*Parametri0[2]
        case 'C':
            TC_x=Parametri0[2]**(-0.33)*1.05*Parametri0[2]
        case 'D':
            TC_x=Parametri0[2]**(-0.5)*1.25*Parametri0[2]
        case 'E':
            TC_x=Parametri0[2]**(-0.4)*1.15*Parametri0[2]

    Tx=2*np.pi*(M/(kel_x*9.81))**0.5
    S_x_eq=F_x_eq/(9.81*Masse)
    axstar=S_x_eq[1]
    dxstar_y=delta_x_eq[1]

    if Tx<TC_x/3:
        Sae_x=Parametri0[0]*S_geo*Parametri0[1]*(Tx/(TC_x/3)+1/Parametri0[1]*(1-Tx/(TC_x/3)))
        Sde_x=Tx**2*Sae_x*9.81/(4*np.pi**2)*10**3
    elif Tx>TC_x/3 and Tx<TC_x:
        Sae_x=Parametri0[0]*S_geo*Parametri0[1]
        Sde_x=Tx**2*Sae_x*9.81/(4*np.pi**2)*10**3
    elif Tx>TC_x and Tx<TD:
        Sae_x=Parametri0[0]*S_geo*Parametri0[1]*TC_x/Tx
        Sde_x=Tx**2*Sae_x*9.81/(4*np.pi**2)*10**3
    elif Tx>TD:
        Sae_x=Parametri0[0]*S_geo*Parametri0[1]*TC_x*TD/Tx**2
        Sde_x=Tx**2*Sae_x*9.81/(4*np.pi**2)*10**3
    
    if (delta_x_eq[2]-Sde_x)>0 and F_x_eq[2]>Sae_x*Masse*9.81:
        dxstar_t=Sde_x
        qstar_x=1
        Saa_x_TR=ADRS_TR[:,1]
        Sda_x_TR=ADRS_TR[:,0]
    else:
        qstar_x=Sae_x/axstar
        if Tx<TC_x:
            dxstar_t=Sde_x/qstar_x*(1+(qstar_x-1)*TC_x/Tx)
        elif Tx>TC_x:
            dxstar_t=Sde_x
        mu_x=dxstar_t/dxstar_y

        R_mu_x = np.zeros(Nt)

        for j in range(Nt):
            if tstep[j]<TC_x:
                R_mu_x[j]=(mu_x-1)*tstep[j]/TC_x+1
            else:
                R_mu_x[j]=mu_x

        Saa_x_TR=ADRS_TR[:,1]/R_mu_x
        Sda_x_TR=mu_x*(ADRS_TR[:,0]/R_mu_x)
    return Tr, ag_TR, ADRS_TR, Sda_x_TR, Saa_x_TR


def _get_current_data(N, D, mu, S, G, V, NZ, alpha, check):
    """ Calculate ductility and drift limits for walls based on input parameters.
    
        Parameters
        ----------
        N : int
            Number of floors.
            
        D : list of np.ndarray
            List of wall dimensions for each floor.
            
        mud : list of np.ndarray
            List of wall friction coefficients for each floor.
            
        S : list of np.ndarray
            List of wall axial forces for each floor.
            
        G : list of np.ndarray
            List of wall material properties for each floor.
        
        V : list of np.ndarray
            List of wall shear forces for each floor.
            
        NZ : list of int    
            List of number of walls for each floor.
            
        alpha : list of np.ndarray
            List of wall orientations for each floor.
            
        check : str
            Type of check to perform ("Ductility check" or "Drift check").

        Returns
        -------
        dult or dult_drift: list of np.ndarray
            List of ductility or drift limits for each floor. """
    
    dult = []
    dult_drift = []

    for k in range(N):
        D0 = D[k]
        S0 = S[k]
        G0 = G[k]
        V0 = V[k]
        dult0 = np.zeros((NZ[k], 2))
        dult_drift0 = np.zeros((NZ[k], 2))
        alpha0 = alpha[k] 
        tau_0 = G0[:, 1]
        a_0 = np.zeros((NZ[k], 2))
        drift_lim = 0.004
        for j in range(2):
            for i in range(NZ[k]):
                b = G0[i, 2]/G0[i, 0]
                a = G0[i, 0]/tau_0[i]
                if alpha0[i] < 45 or alpha0[i] >= 135:
                    dult0[i,j] = mu*((1+S0[i]/(1.5*G0[i,1]))**0.5)/(a/(V0[i]*1.2)*(1/(1+1/(1.2*b)*(V0[i]/D0[i,0])**2)))
                else:
                    if j == 0:
                        dult0[i,j] = mu*((1+S0[i]/(1.5*G0[i,1]))**0.5)/(a/(V0[i]*1.2)*(1/(1+1/(1.2*b)*(V0[i]/D0[i,1])**2)))
                    else:
                        dult0[i,j] = mu*((1+S0[i]/(1.5*G0[i,1]))**0.5)/(a/(V0[i]*1.2)*(1/(1+1/(1.2*b)*(V0[i]/D0[i,0])**2)))

                dult_drift0[i,0]=V0[i]*drift_lim
                dult_drift0[i,1]=V0[i]*drift_lim
        dult.append(dult0)
        dult_drift.append(dult_drift0)
    match check:
        case 'Ductility':
            return dult
        case 'Drift':
            return dult_drift


def pushover_analysis_calculation(EFM, algorithm, method, mu=1.5):
    n_floors, piani, _, alt_s, _, Masse, red_F = EFM.get_floor_data()
    D, X, S, V, alpha, G, NZ = EFM.get_wall_data()
    deltau = _get_current_data(n_floors, D, mu, S, G, V, NZ, alpha, method)

    passo = 5
    incr = 1.001
    deltared=0.2

    ParaTR = EFM.paraTR.values[:, 1:]
    _, TrSLV = _get_Vr_and_Tr(EFM.service_life, EFM.importance_class)
    params = _iterate_return_periods_hazard_params(ParaTR, TrSLV)
    S_Sgeo, Cc = _get_Sgeo_and_Cc(EFM.soil_category, params)
    S_t = _get_S_t(EFM.topographic_category)

    S_geo=S_t*S_Sgeo
    TC=params[2]*Cc

    ADRS, tstep = _get_ADRS(TC, params, S_geo)

    k_ult_TOT, Hult_TOT, vr_ult_TOT, L_organized = _compute_global_vulnerability(D,X,S,V,G,deltau,n_floors,NZ,Masse,alt_s,red_F,ADRS,incr,deltared,piani,alpha,passo,params,ParaTR,tstep,S_geo,TC,EFM.soil_category, algorithm)
    return k_ult_TOT, Hult_TOT, vr_ult_TOT, L_organized

def get_pushover_dataframe(x_coordinates, dxstar_t, Tr, ag_Tr, IR):
    """ Generate dataframe for pushover analysis results.
    
        Parameters
        ----------
        x_coordinates : list of np.ndarray
            List of wall center coordinates for each direction.
            
        dxstar_t : list of float
            List of maximum normalized displacements for each direction.
            
        Tr : list of float
            List of period at maximum displacement for each direction.
            
        ag_Tr : list of float
            List of spectral accelerations at Tr for each direction.
            
        IR : list of float
            List of safety indicies for each direction.
            
        Returns
        -------
        dataframe : pd.DataFrame
            DataFrame containing the pushover analysis results for both directions. """
    
    delta = np.array([x_coordinates[0][1], x_coordinates[1][1]])

    values = np.concatenate((np.array(IR).reshape(-1, 1), np.array(ag_Tr).reshape(-1, 1), np.array(Tr).reshape(-1, 1), delta.reshape(-1, 1), np.array(dxstar_t).reshape(-1, 1)), axis=1)
    values = np.round(values, 2)
    values = np.concatenate((np.array(['X', 'Y']).reshape(-1, 1), values), axis=1)
    columns = ['Direction', 'Safety Index', 'PGA_C', 'TR', 'δ', 'd* (t)']
    return pd.DataFrame(values, columns=columns)

def get_pushover_analysis_results(EFM, k_ult_TOT, Hult_TOT, vr_ult_TOT):
    Masses = np.array([floor.weight for floor in reversed(EFM.floors)]) / 10000

    ParaTR = EFM.paraTR.values[:, 1:]
    _, TrSLV = _get_Vr_and_Tr(EFM.service_life, EFM.importance_class)

    params = _iterate_return_periods_hazard_params(ParaTR, TrSLV)
    S_Sgeo, Cc = _get_Sgeo_and_Cc(EFM.soil_category, params)
    S_t = _get_S_t(EFM.topographic_category)
    S_geo=S_t*S_Sgeo
    TC=params[2]*Cc
    ADRS, tstep = _get_ADRS(TC, params, S_geo)

    x_coordinates, y_coordinates = _calculate_bilinears(vr_ult_TOT, Hult_TOT)
    Saa, Sda, delta_ult_eq, S_eq, dxstar_t, Tr, IR, ADRS_TR, Sda_TR, Saa_TR, ag_Tr = _get_data_for_pushover_results(ADRS, params, Masses, vr_ult_TOT, Hult_TOT, x_coordinates, y_coordinates, k_ult_TOT, tstep, S_geo, TC, ParaTR, EFM.soil_category)
    return ADRS, Sda, Saa, delta_ult_eq, S_eq, dxstar_t, Tr, IR, ADRS_TR, Sda_TR, Saa_TR, ag_Tr

def return_pushover_dataframe(EFM, k_ult_TOT, Hult_TOT, vr_ult_TOT):
    x_coordinates, _ = _calculate_bilinears(vr_ult_TOT, Hult_TOT)
    _, _, _, _, _, dxstar_t, Tr, IR, _, _, _, ag_Tr = get_pushover_analysis_results(EFM, k_ult_TOT, Hult_TOT, vr_ult_TOT)
    df = get_pushover_dataframe(x_coordinates, dxstar_t, Tr, ag_Tr, IR)
    return df

def plot_capacity_curve(vr_ult_TOT, Hult_TOT):
    """ Generate a plot comparing pushover capacity curves with their bilinear approximations.

        Parameters
        ----------
        vr_ult_TOT : list of np.ndarray
            List of ultimate displacements for X and Y directions.

        Hult_TOT : list of np.ndarray
            List of ultimate base shear forces for X and Y directions.  
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object containing the pushover plot. """
    
    x_coordinates, y_coordinates = _calculate_bilinears(vr_ult_TOT, Hult_TOT)
    fig = plot_utils._plot_capacity_curve(vr_ult_TOT, Hult_TOT, x_coordinates, y_coordinates)
    return fig


def plot_ADRS_curve(EFM, k_ult_TOT, Hult_TOT, vr_ult_TOT):
    ADRS, Sda, Saa, delta_ult_eq, S_eq, dxstar_t, Tr, IR, ADRS_TR, Sda_TR, Saa_TR, _ = get_pushover_analysis_results(EFM, k_ult_TOT, Hult_TOT, vr_ult_TOT)
    fig = plot_utils._plot_ADRS(ADRS, Sda, Saa, delta_ult_eq, S_eq, dxstar_t, Tr, IR, ADRS_TR, Sda_TR, Saa_TR)
    return fig