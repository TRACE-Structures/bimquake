import numpy as np
import math
from copy import deepcopy
import pandas as pd
from .seismic_loading import iterate_return_periods_ADRS


############################################################################
def get_force_displacement_diagram(D,X,S,V,G,
                                  deltau,
                                  N,NZ,
                                  Masses,
                                  red_F,
                                  inc,
                                  deltared,
                                  alpha,
                                  passo,
                                  algorithm):
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
        k_ult_TOT : tuple
            Tuple containing ultimate stiffness values in x and y directions.
        
        Hult_TOT : tuple
            Tuple containing ultimate horizontal force values in x and y directions.

        vr_ult_TOT : tuple
            Tuple containing ultimate displacement values in x and y directions.

        L : np.ndarray
            Array containing information about collapsed walls."""

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


def get_bilinear_points_coord(X, Y):
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


def compute_seismic_performance_assesment(design_params, Masses, v, H, K, delta_ult_eq, Hult_eq, soil_category):
    """ Helper function to compute data for plotting pushover curves and response spectra.

        Parameters
        ----------
        design_params : dict
            Dictionary containing design parameters and seismic data.
            
        Masses : np.ndarray
            Array of masses for each floor.

        v : list of np.ndarray
            List of displacements for X and Y directions.

        H : list of np.ndarray
            List of base shear forces for X and Y directions.

        K : list of np.ndarray
            List of stiffness values for X and Y directions.

        delta_ult_eq : list of np.ndarray
            List of ultimate equivalent displacements for X and Y directions.

        Hult_eq : list of np.ndarray
            List of ultimate equivalent base shear forces for X and Y directions.

        soil_category : str
            Soil category for the analysis.
            
        Returns
        -------
        pushover_results : dict
            Dictionary containing results for pushover analysis and seismic performance assessment. """
    
    spectral_func = design_params["spectral_func"]
    t_step = design_params["tstep"]
    ADRS = design_params["ADRS"]
    TC =  design_params["TC"]
    S_geo = design_params["S_geo"]
    ag_SLV = design_params["ag_SLV"]

    MTOT = np.sum(Masses)

    Saa, Sda, S_eq, dxstars, k_el_TOT = [], [], [], [], []

    for i in range(2):  # X and Y
        e_idx = _get_elastic_index(K[i])
        
        k_el = (H[i][e_idx] / 9.81) / (v[i][e_idx] / 1000)
        k_el_TOT.append(k_el)
        
        T = 2 * np.pi * np.sqrt(MTOT / (k_el * 9.81))

        S_eq_i = Hult_eq[i] / (9.81 * MTOT)
        S_eq.append(S_eq_i)

        axstar = S_eq_i[1]
        dxstar_y = delta_ult_eq[i][1]

        Sae, Sde = spectral_func(T)
        # Sde = T**2 * Sae * 9.81 / (4 * np.pi**2) * 1e3

        if (delta_ult_eq[i][2] - Sde) > 0 and Hult_eq[i][2] > Sae * MTOT * 9.81:
            dxstar = Sde
            Saa_i = ADRS[:, 1]
            Sda_i = ADRS[:, 0]
        else:
            q = Sae / axstar

            if T < TC:
                dxstar = Sde / q * (1 + (q - 1) * TC / T)
            else:
                dxstar = Sde

            mu = dxstar / dxstar_y

            R_mu = np.ones_like(t_step)
            mask = t_step < TC
            R_mu[mask] = (mu - 1) * t_step[mask] / TC + 1
            R_mu[~mask] = mu

            Saa_i = ADRS[:, 1] / R_mu
            Sda_i = mu * (ADRS[:, 0] / R_mu)

        Saa.append(Saa_i)
        Sda.append(Sda_i)
        dxstars.append(dxstar)
      

    Tr_x, ag_x_TR, ADRS_x_TR, Sda_x_TR, Saa_x_TR = iterate_return_periods_ADRS(
        MTOT, Hult_eq[0], design_params, delta_ult_eq[0],
        k_el_TOT[0], soil_category
    )

    Tr_y, ag_y_TR, ADRS_y_TR, Sda_y_TR, Saa_y_TR = iterate_return_periods_ADRS(
        MTOT, Hult_eq[1], design_params, delta_ult_eq[1],
        k_el_TOT[1], soil_category
    )

    Tr = [Tr_x, Tr_y]
    ADRS_TR = [ADRS_x_TR, ADRS_y_TR]
    Sda_TR = [Sda_x_TR, Sda_y_TR]
    Saa_TR = [Saa_x_TR, Saa_y_TR]

    IR = [
        round(ag_x_TR / ag_SLV * 100) / 100,
        round(ag_y_TR / ag_SLV * 100) / 100
    ]

    ag_TR = [ag_x_TR, ag_y_TR]

    pushover_results =  {
        "Saa": Saa,   # pushover-derived capacity/demand curve - acceleration
        "Sda": Sda,   # pushover-derived capacity/demand curve - displacement
        "S_eq": S_eq,  # Equivalent seismic demand vector per floor direction
        "dxstars": dxstars, # Target displacement (performance point candidates), Displacement obtained from intersection with demand spectrum
        "Tr": Tr,  # Return period (years)  Earthquake intensity level that best matches structural capacity
        "IR": IR,  # Intensity ratio ag_TR/ag_ref
        "ADRS_TR": ADRS_TR, # Acceleration–Displacement Response Spectrum matrix, first column disp, second acceleration
        "Saa_TR": Saa_TR, # Acceleration demand after TR scaling
        # Displacement demand after TR scaling
        "Sda_TR": Sda_TR,
        # Peak ground acceleration (PGA) at selected return period:
        "ag_TR": ag_TR,  # Ground shaking intensity corresponding to Tr
        "ADRS": ADRS, # Original acceleration–displacement response spectrum
        # Periods at which ADRS is evaluated
        "t_step": t_step,
        "S_geo": S_geo }
        
    return pushover_results
          

def _get_elastic_index(k):
    """ Get the index of the first non-zero element in the array k.
    
        Parameters
        ----------
        k : list or np.ndarray
            Array of stiffness values.
            
        Returns
        -------
        index : int
            Index of the first non-zero element in k. """
    
    diff = np.diff(k)
    idx = np.where(diff != 0)[0]
    index = len(k) - len(idx)
    return index

def get_current_data(N, D, mu, S, G, V, NZ, alpha, check):
    """ Calculate ductility and drift limits for walls based on input parameters.
    
        Parameters
        ----------
        N : int
            Number of floors.
            
        D : list of np.ndarray
            List of wall dimensions for each floor.

        mu : float
            Ductility factor for the walls.

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
        # a_0 = np.zeros((NZ[k], 2))
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


def get_global_vulnerability_metrics(v_bl, pushover_resuls):
    """ Calculate global vulnerability metrics based on pushover analysis results.
    
        Parameters
        ----------
        v_bl : list of tuples
            List of tuples containing base shear and displacement for X and Y directions.
            
        pushover_resuls : dict
            Dictionary containing results from pushover analysis.
            
        Returns
        -------
        df : pd.DataFrame
            DataFrame containing global vulnerability metrics including Safety Index, PGA, Return Period, Drift, and Estimated Peak Displacement."""
    
    # Estimated peak displacement of the structure under seismic demand                    
    dxstar_t = pushover_resuls["dxstars"]  # intersection with demand spectrum
    # Return period - Earthquake intensity level that best matches structural capacity
    Tr = pushover_resuls["Tr"]
    # Peak ground acceleration (PGA) at selected return period
    ag_Tr = pushover_resuls["ag_TR"]
    # Intensity ratio - How strong the selected seismic demand is compared to reference hazard
    IR = pushover_resuls["IR"]
    
    delta = np.array([v_bl[0][1], v_bl[1][1]])

    values = np.concatenate((np.array(IR).reshape(-1, 1), np.array(ag_Tr).reshape(-1, 1), np.array(Tr).reshape(-1, 1), delta.reshape(-1, 1), np.array(dxstar_t).reshape(-1, 1)), axis=1)
    values = np.round(values, 2)
    values = np.concatenate((np.array(['X', 'Y']).reshape(-1, 1), values), axis=1)
    columns = ['Direction', 'Safety Index', 'PGA_C', 'TR', 'δ', 'd* (t)']
    df = pd.DataFrame(values, columns=columns)
    return df