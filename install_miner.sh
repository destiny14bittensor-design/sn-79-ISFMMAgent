#!/bin/sh
set -e
echo $PATH
echo 'apt update'
apt-get update

if ! command -v pm2
then
	if ! command -v nvm
	then
		echo 'Installing nvm...'
		curl -o install_nvm.sh https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh
		chmod +x install_nvm.sh && ./install_nvm.sh
		export NVM_DIR="$HOME/.nvm"
		[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
		[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
		nvm install node		
		echo export PATH="'$NVM_BIN:$PATH'" >> ~/.bashrc;
		export PATH=$NVM_BIN:$PATH
		rm install_nvm.sh
	fi
	echo 'Installing pm2...'
	npm install --location=global pm2
	pm2 install pm2-logrotate
	pm2 set pm2-logrotate:max_size 100M
	pm2 set pm2-logrotate:compress true
fi

if ! command -v htop
then
	echo 'Installing htop...'
	apt-get install htop -y
fi

if ! command -v tmux
then
	echo 'Installing tmux...'
	apt-get install tmux
fi

if ! command -v pyenv
then
    echo 'Installing pyenv'
    sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
    curl https://pyenv.run | bash
	export PYENV_ROOT="$HOME/.pyenv"
	export PATH="$PYENV_ROOT/bin:$PATH"
	eval "$(pyenv init --path)"
	eval "$(pyenv init -)"
    echo 'export PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init --path)"\neval "$(pyenv init -)"' >> ~/.bashrc
fi
if ! pyenv versions | grep -Fq '3.10.9';
then
    echo 'Installing and activating Python 3.10.9'
    pyenv install 3.10.9
    pyenv global 3.10.9
fi

python -m pip install -U pyopenssl cryptography

echo "Installing taos"
python -m pip install -e .
mkdir -p ~/.taos
cp -r agents ~/.taos/agents