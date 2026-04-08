-- data/schema.sql — Table definitions for the simulated e-commerce database
--
-- Four tables modelling a small e-commerce company:
--   customers, products, orders, order_items
--
-- This file is executed by seed.py to create the empty tables before
-- populating them with synthetic data.

CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT UNIQUE NOT NULL,
    signup_date DATE NOT NULL,
    region      TEXT NOT NULL  -- values: North, South, East, West
);

CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    category   TEXT NOT NULL,  -- values: Electronics, Apparel, Home, Sports, Beauty
    price      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date  DATE NOT NULL,
    total_amount REAL NOT NULL,
    status      TEXT NOT NULL  -- values: completed, refunded, pending
);

CREATE TABLE IF NOT EXISTS order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL,
    unit_price    REAL NOT NULL
);
