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


'''
COMPARACIÓ EULER EXPLICIT
'''
#es defineix els tres grafics per Euler explicit en un plot per poder comparar-los
L_real = 2 # L'amplada de la paret són 2 cm.
t_real = t_a * t0

valors_x = L_real * valors_xbar # Eliminem l'adiamensionalització de la coordenada espaial.

'''
SOLUCIO ANALITICA
'''
T_analitica = solucio_anal(valors_xbar, t_a, 1000)

'''
EULER EXPLICIT
'''
T_EE_051 = euler_explicit(0.51)
T_EE_049 = euler_explicit(0.49)
T_EE_025 = euler_explicit(0.25)

'''
GRAFIQUES
'''

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[1].set_title(f"(r = 0.49) Estable")
ax[1].plot(valors_x, Tc + T0*T_EE_049[-1,:], label = "(r = 0.49) Estable", color ="blue", alpha = 0.6)
ax[1].plot(valors_x, T_analitica, color = "red", linestyle = "--", label = "Solució analítica")
ax[1].grid(alpha=0.2)
ax[1].set_xlabel("Posició (cm)")
ax[1].set_ylabel("Temperatura (ºC)")
ax[1].axhline(50, color="black", linestyle=":", label="Llindar de 50 °C")
ax[1].axvspan(3*L_real/8, 5*L_real/8, color = "red", alpha = 0.15, label = "Teixit malalt")
ax[1].legend()

ax[2].set_title(f"(r = 0.25) Estable")
ax[2].plot(valors_x, Tc + T0*T_EE_025[-1,:], label = "Euler explícit (r = 0.25)", color = "blue", alpha = 0.6)
ax[2].plot(valors_x, T_analitica, color = "red", linestyle = "--", label = "Solució analítica")
ax[2].grid(alpha=0.2)
ax[2].set_xlabel("Posició (cm)")
ax[2].set_ylabel("Temperatura (ºC)")
ax[2].axhline(50, color="black", linestyle=":")
ax[2].axvspan(3*L_real/8, 5*L_real/8, color = "red", alpha = 0.15)
ax[2].legend()

ax[0].set_title(f"(r = 0.51) Inestable")
ax[0].plot(valors_x, Tc + T0*T_EE_051[-1,:], label = "Euler explícit (r = 0.51)", color ="blue", alpha = 0.6)
ax[0].plot(valors_x, T_analitica, color = "red", linestyle = "--", label = "Solució analítica")
ax[0].grid(alpha=0.2)
ax[0].set_xlabel("Posició (cm)")
ax[0].set_ylabel("Temperatura (ºC)")
# ax[0].axhline(50, color="black", linestyle=":", label="Llindar de 50 °C")
ax[0].axvspan(3*L_real/8, 5*L_real/8, color = "red", alpha = 0.15)
ax[0].legend()

plt.show()