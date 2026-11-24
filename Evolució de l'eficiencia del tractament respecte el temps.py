import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

'''
Escrivim les constant que necessitarem pel nostre problema.
'''
rho = 1081
cv = 3698
kappa = 0.56
sigma = 0.472
L_tot = 0.02
L_zona_malalta = 0.005
V_eff = 40/np.sqrt(2)
Tc = 36.5

'''
variables q calculem o definim per adimensionalitzar.
'''
Pext = sigma*(V_eff/L_tot)**2
alpha = kappa/(cv*rho)
x0 = L_tot
t0 = L_tot**2/alpha
T0 = sigma*V_eff**2/kappa

print(f"Difusivitat térmica (alpha): {alpha:.4e} m^2/s")
print(f"Temps característic (t0): {t0:.2f} s ({t0/60:.2f} min)")
print(f"Temperatura característica (T0): {T0:.2f} °C\n")

N=101                             # nombre de punts que fare servir (101-1 = 100 intervals)
L = 1                             # Longitud del segment adimensionalitzat.
valors_xbar=np.linspace(0, L, N)  # N valors repartits uniformement entre 0 i 1
delta_xbar=L/(N-1)                # distància entre 2 punts consecutius
t_a = 0.025                         # temps objectiu segons l'enunicat


'''
SOLUCIO ANALITICA
'''

def solucio_anal(x, t, nombre_termes_sumatori):
  Tbar=np.zeros_like(x, dtype="float") #vector/matriu ple de zeros amb la mateixa mida i forma que x, dtype important o sino python potser es pensa q treballem amb ints.
  #definim la matriu amb 0's perque es mes facil de definir una matriu nula que una amb Tc a tot arreu, sumarem Tc mes tard per complir les condicions inicials/contorn
  for k in range(nombre_termes_sumatori):
    n = 2*k+1
    Tbar += (4 * T0 / (n * np.pi)**3) * ( 1 - np.exp( - ((n * np.pi)**2) * t)) * np.sin(n * np.pi * x) # Equació (1) de l'informe sense TC (ho sumem quan fem el return).
  return Tbar + Tc



'''
SOLUCIO AMB METODE EULER EXPLICIT
'''

def euler_explicit(valor_r):
  delta_tbar = valor_r*delta_xbar**2          # Relació pels passos espaials i temporals que no són independents (r es el valor que apareix al guio).
  passes = int(np.round(t_a/delta_tbar))      # Enter que indica el nombre de passos fets.
  delta_tbar = t_a/passes                     # Tornem a redefinir \Delta \bar{t} per tal d'acabar en el temps t_a exactament.
  new_r = delta_tbar/delta_xbar**2            # Definim la nova r.
  T_bar=np.zeros(N)                           # Llista de la temperatura a l'inici que es troba tot a 0, i després ja li sumarem la temperatura Tc per complir condicions de contorn.
  matriu_T = np.zeros((passes+1, N))          # Creem una matriu amb [passes+1] files i N columnes, on guardarem les Temperatures de cada iteració.

  '''
  En el següent "for" definim el mètode iteratiu pel mètode de Euler. En aquest cas, creem una còpia de la llista de temperatures a tot l'espai en l'instant
  anterior per poder-lo modificar durant l'iteració. Definim la temperatura al següent pas tal com es mostra a l'Equació (2.1) de l'informe. Com que estem
  calculant la temperatura en el següent instant (n+1) i al iterar només depenen de l'instant (n) els elements necessaris, per tant durant el mateix pas podrem
  calculat el valor de la temperatura per a tots els punts en l'instant (n+1). A més a més, els valors dels extrems han de coincidir amb les condicions de
  contorn establertes, és a dir que han de coincidir amb Tc.
  '''

  for i in range(passes):
    T_seguent_pas = T_bar.copy()                  # Fem un copia de les temperatures del pas anterior.
    T_seguent_pas[1:-1] = ((1-new_r*2)*T_bar[1:-1] # Modifiquem el valor des de [1] fins a [N-1]. Els contorns no es poden modificar tant [0] com [N] ja que es mantenen a temperatura constant.
                          + new_r*(T_bar[2:]      # Recòrrer la llista de T_bar des de [2] fins a [N].
                          + T_bar[:-2])           # Pren els valors de la llista de T_bar des de [0] fins a [N-2]
                          + delta_tbar)            # Sumem el terme de \Delta_\bar{t}
    '''
    Anem a mirar quins elements de les lliste agafa cada element del mètode d'iteració
    T_bar[1:-1] -> [1,2,...,N-2,N-1] -> j
    T_bar[2:]   -> [2,3,...,N-1, N]  -> j+1
    T_bar[:-2]  -> [0,1,...,N-3,N-2] -> j-1
    Considerem només els elements per a j en [1,N-1] evitant així les condicions de contorn fixades al llarg del procés.
    '''
    T_bar = T_seguent_pas                # Copiem la llista amb els valors modificats de T per fer la pròxima iteració.
    matriu_T[i+1, :] = T_seguent_pas     # Modifiquem a partir de la segona columna [1] perquè a la primera [0] tindrem una columna de 0.
  return matriu_T                        # Com a output tindrem una matriu amb les temperatures a cada pas.


'''
SOLUCIO AMB METODE EULER IMPLICIT
'''

def euler_implicit(valor_r, tolerancia, max_passes):
  delta_tbar = valor_r*delta_xbar**2          # Relació pels passos espaials i temporals que no són independents (r es el valor que apareix al guio).
  passes = int(np.round(t_a/delta_tbar))      # Enter que indica el nombre de passos fets.
  delta_tbar = t_a/passes                     # Tornem a redefinir \Delta \bar{t} per tal d'acabar en el temps t_a.
  new_r = delta_tbar/delta_xbar**2            # Definim la nova r.
  T_bar = np.zeros(N)                         # Llista de la temperatura a l'inici que es troba tot a 0, i després ja li sumarem la temperatura Tc.
  matriu_T = np.zeros((passes+1, N))

  '''
  Utilitzarem el mètode de GAUSS-SEIDEL per solucionar el problema. Per poder aplicar aquest mètode, haurem d'establir una tolerància i així poder truncar
  el procés. En aquest cas, la tolerància s'entra com a input al cridar la funció.
  '''
  for i in range(passes):
    T_seguent_pas = T_bar.copy()     # Fem un copia de les temperatures del pas anterior.
    for _ in range(max_passes):      # El procés es repeteix max_passes per truncar el procés si no s'arriba a obtenir un error per sota la tolerància.
      error_max=0.0                  # ens servirà per anotar el canvi més gran que ha patit algun punt de la barra durant aquesta passada
      for j in range(1, N-1):        #important, de 1 a N-1 perque els bordes de T barra = 0 sempre per cond contorn
        valor_antic = T_seguent_pas[j] #ens guardem el número que havia en aquest punt abans de recalcular-lo
        T_seguent_pas[j] = (T_bar[j] + delta_tbar + new_r*(T_seguent_pas[j+1]+T_seguent_pas[j-1]))/(1.0+2.0*new_r)
        diferencia = abs(T_seguent_pas[j]-valor_antic)

        if diferencia > error_max:
          error_max = diferencia
      if error_max < tolerancia:
          break

    T_bar = T_seguent_pas
    matriu_T[i+1, :] = T_seguent_pas

  return matriu_T




'''
SOLUCIO AMB METODE CRANK-NICOLSON
'''
def Crank_Nicolson (valor_r, tolerancia, max_passes):
  delta_tbar=valor_r*delta_xbar**2          # Relació pels passos espaials i temporals que no són independents (r es el valor que apareix al guio).
  passes=int(np.round(t_a/delta_tbar))      # Enter que indica el nombre de passos fets.
  delta_tbar = t_a/passes                   # Tornem a redefinir \Delta \bar{t} per tal d'acabar en el temps t_a.
  new_r = delta_tbar/delta_xbar**2          # Definim la nova r.
  T_bar=np.zeros(N)                         # Llista de la temperatura a l'inici que es troba tot a 0, i després ja li sumarem la temperatura Tc.

  matriu_T = np.zeros((passes+1, N))
  matriu_T[0, :] = T_bar

  for i in range(passes):
    T_seguent_pas = T_bar.copy()            # Fem un copia de les temperatures del pas anterior.

    '''
    Utilitzarem el mètode de GAUSS-SEIDEL per solucionar el problema. Per poder aplicar aquest mètode, haurem d'establir una tolerància i així poder truncar
    el procés. En aquest cas, la tolerància s'entra com a input al cridar la funció.
    '''
    for _ in range(max_passes):
      error_max=0.0 #ens servirà per anotar el canvi més gran que ha patit algun punt de la barra durant aquesta passada
      for j in range(1, N-1): # El procés es repeteix max_passes per truncar el procés si no s'arriba a obtenir un error per sota la tolerància.
        valor_antic = T_seguent_pas[j]
        T_seguent_pas[j] = (T_bar[j] + delta_tbar + (new_r/2) * (T_seguent_pas[j+1] + T_seguent_pas[j-1] + T_bar[j+1] - 2*T_bar[j] + T_bar[j-1])) / (1+new_r)
        diferencia = abs(T_seguent_pas[j] - valor_antic)     # Calculem la diferència per poder parar el procés posteriorment si s'assoleix certa tolerància.

        if diferencia >= error_max:
          error_max = diferencia                             # Si la diferència entre el valor del pas següent i l'actual és superior o inferior a l'error continuem                                                     # el procés, però considerem l'error màxim pel pròxim pas com la diferència de l'anterior.
      if error_max < tolerancia:
        break                                                # Trenquem el bucle si s'assoleix un error inferior a la tolerància.

    T_bar = T_seguent_pas
    matriu_T[i+1, :] = T_seguent_pas
  return matriu_T


#Mirarem com es calenta la part malalta i sana.
#basicament, Veurem cuanta part sana es fa malbe si deixem el voltatge de 40V mes (o menys) estona
#ho avaluare de 0 a 0,1 de t barra (que equival a uns cuants minuts en temps normal) aixi podem veure quan hauriem de parar per tal de no fer malbe la part sana pero tenir la malalta al rang q toca durant el maxim de temps
t_eval = np.linspace(0, 0.1, 400)
t_seg = t0*t_eval

valors_x_augmentats = np.linspace(0.0, 1.0, 1001)
part_malalta = (valors_x_augmentats >= 0.375) & (valors_x_augmentats <= 0.625) #aixo es una llista de "trues" i "fals", basicament va "fals, fals, fals, ..." fins arribar a x=0375, llavors "true, true..." i torna a fals quan pasem de 0.625
part_sana = ~part_malalta #el signe ~basicament agafa la llsita d'abans i la "inverteix" (tots els trues son fals i fals true), llavors la llista fa true, true, true... fals, fals..., true... per la part sana!

T_centre=[]
T_sana_punt = [] # en aquesta llista avaluarem l'evolucio de la temperatura en un punt del teixit sa
pct_malalta_ok = []     # % de teixit malalt entre 50 °C y 80 °C
pct_sana_feta_malbe = []    # % de teixit sa que supera els 50 °C
T_max_global = []

#fare servir la solucio analitica per analitzar aquest apartat. T_perfil sera la temperatura de perfil de tot el teixit. T_centre la del centre en concret (per veure l'evolucio de temperatura de la zona malalta, ja que al centre es on mes puja)
#I T_sana_punt es la temperatura de la zona sana en un punt random (he escollit x barra=0.25 perque esta molt centrat a la zona sana, pero si volem veure el cas més extrem hauriem d'agafar el borde, xbar=0.375 aprox)
for t in t_eval:
    T_perfil = solucio_anal(valors_x_augmentats, t, 1000)
    T_centre.append(solucio_anal(np.array([0.5]), t, 1000)[0])
    T_sana_punt.append(solucio_anal(np.array([0.25]), t, 1000)[0])
    T_max_global.append(np.max(T_perfil))

    malalta_ideal = (T_perfil[part_malalta] >= 50.0) & (T_perfil[part_malalta] < 80.0)
    sana_supera_50 = T_perfil[part_sana] >= 50.0

    pct_malalta_ok.append(100.0 * np.mean(malalta_ideal)) # % de la zona malalta q esta en el rang que toca de 50-80 graus
    pct_sana_feta_malbe.append(100.0 * np.mean(sana_supera_50))# % de la zona sana q esta en el rang que no toca, pasant de 50 graus

plt.figsize=(13, 5)

#aquest grafic es innecessari per l'informe, pero es interesant veure com evoluciona termodinamicament els punts que hem escollit
plt.plot(t_seg, T_centre, color="red", linestyle="--", lw=1.8, label="Zona malalta al centre (x = 1.00 cm)")
plt.plot(t_seg, T_sana_punt, color="green", linestyle="--", lw=2, label="Zona sana en el punt x = 0.50 cm")
plt.hlines(50.0, 0, t0*0.1, color='orange', linestyle=':', lw=1.8, label="50 °C")
plt.hlines(80.0, 0, t0*0.1, color='darkred', linestyle=':', lw=1.8, label="80 °C")
plt.title("Temperatura en la zona malalta vs en la zona sana (40 V)")
plt.xlabel("Temps t [s]")
plt.ylabel("Temperatura T [°C]")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#grafic dels % de teixit sa destruit i teixit malalt al rang correcte, veiem que ta es una bona estimacio de quan hauriem de parar el tractament (considerant que el teixit sa no es fa malbe immediatament al assolir una T>50ºC)
valor_t_en_ta=t_a*t0
print(valor_t_en_ta)

plt.figsize=(13, 5)
plt.plot(t_seg, pct_malalta_ok, color="salmon", linestyle="--", lw=2, label="% zona malalta en el rang desitjat [50 °C, 80 °C)")
plt.plot(t_seg, pct_sana_feta_malbe, color="crimson", linestyle="--", lw=2, label="% zona sana amb T >= 50 °C")
plt.vlines(t_a*t0, 0, 105, color="darkblue", linestyle=":", lw=2, label=r"$t=t_{a}·t_{0}=71.4s$")
plt.hlines(50, 0, 0.1*t0, color="darkgreen", linestyle=":", lw=2, label="50%")
plt.title("Eficiencia del tractament respecte el temps")
plt.xlabel("Temps t [s]")
plt.ylabel("Percentatges de la regió [%]")
plt.ylim(-2, 105)
plt.legend(fontsize=8, loc="lower center")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()