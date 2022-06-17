from ex.server import app

version = '0.1.0'
if __name__ == '__main__' :
    
    print('------------------------------------------------')
    print('Wandlab CV - version ' + version )
    print('------------------------------------------------')
    
    app.run(host='127.0.0.1', port=6000 )