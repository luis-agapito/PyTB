Ef=1.828967854241443E-001*2 
set yrange [-15:6]
#plot './silicon.dat.gnu' u 1:($2-Ef)*13.6 w l lc rgb 'green', '../test2_out/bands_ortho_ws.txt' u 1:2 w l lc rgb 'black'
plot 'silicon.dat.gnu' u 1:($2-Ef)*13.6 w l lc rgb 'green', '../test_out/bands_nonortho_ws.txt' u 1:2 w l lc rgb 'black'
