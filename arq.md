# 🗺️ Arquitectura de Información y Sitemap – Sistema de Cine

El presente sitemap organiza los módulos de la aplicación divididos en dos entornos: el **Portal del Cliente** (orientado a la consulta y compra) y el **Panel de Administración** (destinado a la gestión integral de entidades y parametrizaciones del sistema).

---

## 1. Diagrama General del Sistema (Mermaid)

```mermaid
graph TD
    App[🎬 Sistema de Cine]

    %% Módulos Principales
    App --> Publico[🌐 Portal Público / Cliente]
    App --> Admin[🛠️ Panel de Administración / Backoffice]

    %% Portal Cliente
    Publico --> Home[🏠 Inicio / Cartelera]
    Publico --> Peliculas[🎬 Películas & Sinopsis]
    Publico --> Horarios[🕒 Listado de Horarios por Tipo de Función]
    Publico --> Promos[🏷️ Listado de Promociones por Día]
    Publico --> Compra[🎟️ Flujo de Compra de Entrada]
    Publico --> Perfil[👤 Mi Cuenta / Mis Entradas]

    %% Detalle Película
    Peliculas --> FichaPelicula[Ficha Técnica & Sinopsis]
    FichaPelicula --> ModuloResenas[⭐ Reseñas y Calificaciones]

    %% Flujo de Compra
    Compra --> Paso1_Funcion[1. Selección de Función y Horario]
    Paso1_Funcion --> Paso2_Butacas[2. Selección de Butacas en Sala]
    Paso2_Butacas --> Paso3_Entradas[3. Selección de Tipo de Entrada & Promociones]
    Paso3_Entradas --> Paso4_Checkout[4. Checkout & Pasarela de Pago MP / Stripe]
    Paso4_Checkout --> Paso5_Confirmacion[5. Confirmación & Envío de Entrada por Email]

    %% Panel Admin - CRUDs
    Admin --> Dashboard[📊 Dashboard / Resumen General]
    Admin --> ModUsuarios[👥 Gestión de Usuarios - CRUD]
    Admin --> ModCartelera[🎬 Gestión de Películas - CRUD]
    Admin --> ModSalas[🏛️ Gestión de Salas - CRUD]
    Admin --> ModProgramacion[📅 Programación de Funciones]
    Admin --> ModPrecios[💵 Tarifas y Valores Históricos]
    Admin --> ModVentas[💳 Ventas y Entradas - CRUD]
    Admin --> ModPromosAdmin[🎁 Gestión de Promociones]

    %% Detalle Admin
    ModProgramacion --> CrudFunciones[CRUD Función]
    ModProgramacion --> CrudTipoFuncion[CRUD Tipo de Función]
    ModPrecios --> CrudTipoEntrada[CRUD Tipo de Entrada]
    ModPrecios --> CrudValorHistorico[CRUD Valor Histórico]
    ModVentas --> CrudVentas[CRUD Venta]
    ModVentas --> CrudEntradas[CRUD Entrada]