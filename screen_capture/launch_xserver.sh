


Xvfb :1 -screen 0 800x600x24 -pixdepths 3 -fp /usr/share/fonts/X11/Type1 &

# compile and launch screen capture
sh compile.sh
DISPLAY=:1 && ./capture_and_write &
DISPLAY=:1 && steam steam://rungameid/311690 &

# si parametro 1 es 1 inicia con visualizador
if [ $1 -eq 1 ]
then
	echo "Iniciando con visualizacion"
	x11vnc -display :1 -bg -nopw -listen localhost -xkb &
	vncviewer :0
fi


