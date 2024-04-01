# sanal ortam : farklı kütüphane ve versiyon ihtiyaçlarını çalışmalar
# birbirini etkilemeyecek şekilde oluşturma imkanı sağlar. (conda)

# paket yönetimi : pip,conda(hem sanal ortam hem de paket yöneticisi) paket yönetimi ve bağımlılık yönetimi(dependecy management)
# pip ve conda beraber kullanılır.

# Sanal Ortam ve Paket Yönetimi
# (terminalde yapılır bu işlemler)

# sanal ortamların listelenmesi
# conda env list (yanında yıldız olan içinde olduğumuz ortam)

# sanal ortam oluşturma
# conda create -n myenv(istenen isim buraya)

# sanal ortamı aktif etme
# conda activate myenv

# yüklü paketlerin listelenmesi
# conda list

# paket yükleme
# conda install numpy (paketi yüklerken bağımlılıklar da gelir)

# birden fazla paketi aynı anda yükleme
# conda install numpy pandas scipy

# paket silme
# conda remove package_name

# bir paketin farklı bir versiyonunu yükleme
# conda install numpy=1.20.1

# paket yükseltme
# conda upgrade numpy ya da tüm paketleri yükseltmek için conda upgrade -all

# pip: pypi(python package index) paket yönetim aracı
# pip ile paket yükleme : pip install pandas
# versiyona göre paket yükelme : pip install pandas==1.2.1

# paket versiyonları ve paketleri yaml dosyası haline getirme
# conda env export > environment.yaml
# pip ile de yapılabilir pip ile requirements.txt conda ile environment.yaml yapılır.

# environment silme
# önce ortam içindeysek conda deactivate ile çıkmamız sonra silmemiz gerekiyor.
# conda env remove -n myenv

# elimizde var olan yaml dosyası ile oradaki paket ve versiyonları yükleme
# conda env create -f environment.yaml
# daha sonra conda activate myenv ile tekrar aktif etmemiz gerekiyor.