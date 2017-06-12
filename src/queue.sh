while [ -n "$1" ]
do
  case "$1" in
  -y) param="$2"
  year=$param
  shift ;;
  -b) param="$2"
  start_month=$param
  shift ;;
  -e) param="$2"
  end_month=$param
  shift ;;
  --) shift
  break ;;
  *) echo "$1 is not an option";;
  esac
shift
done

for (( i=$start_month; i<=$end_month; i++ ))
do
  now="$(date +"%d.%m.%Y[%H:%M:%S]")"
  echo "Script started at $now"
  ~/anaconda/bin/python loader.py  -s cond-mat -y $year -m $i
done
