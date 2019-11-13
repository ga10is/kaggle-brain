import torch
import os
import sqlite3
from contextlib import closing
import textwrap
import json
import pandas as pd
import numpy as np
import datetime

try:
    import matplotlib.pylab as plt
except ImportError:
    print('Unable import matplotlib')


def create_table(dbfile):
    with closing(sqlite3.connect(dbfile)) as con:
        cursor = con.cursor()
        # create experiment table
        sql = textwrap.dedent('''\
            create table experiment (
            exid integer primary key autoincrement,
            time timestamp,
            name text,
            arch text,
            path text
            )
            ''')
        print(sql)
        cursor.execute(sql)

        # create history table
        sql = textwrap.dedent('''\
            create table history (
            hid integer primary key autoincrement,
            time timestamp,
            exid integer,
            epoch integer,
            iter integer,
            mode text,
            batch_size integer,
            lr real,
            loss real,
            metrics real
            )
            ''')
        print(sql)
        cursor.execute(sql)

        sql = textwrap.dedent('''\
            create table grad (
            gid integer primary key autoincrement,
            exid integer,
            epoch integer,
            iter integer,
            layer_grad blob
            )
            ''')
        print(sql)
        cursor.execute(sql)

        con.commit()


class ModelDB:
    """

    Notes
    -----
    The database has 2 tables.
    - experiment table: the table which records model name, architecture and model path.
    - history table: the table which records loss, metrics, lr, batch size, etc. for each iteration.
    """
    time_setting = sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES

    def __init__(self, dbfile):
        """
        Initialize instance.

        Paramters
        ---------
        dbfile: str
            file path of sqlite
        model: torch.nn.Module
        """
        if not os.path.exists(dbfile):
            raise ValueError('DB file not found: %s' % dbfile)

        self.dbfile = dbfile
        # experiment id: primary key of experiment table
        self.ex_id = 'no_exid'

    def rec_model(self, model, model_path):
        """
        write model information to experiment table.

        Paramters
        ---------
        model: torch.nn.Module
        """
        with closing(sqlite3.connect(self.dbfile, detect_types=self.time_setting)) as con:
            cursor = con.cursor()

            sql = 'insert into experiment(time, name, arch, path) values(?, ?, ?, ?)'
            model_name = model.__class__.__name__
            architecture = str(model)
            # self.model_id = hashlib.md5(architecture.encode()).hexdigest()
            now = datetime.datetime.now()
            data = (now, model_name, architecture, model_path)

            cursor.execute(sql, data)
            self.ex_id = cursor.lastrowid

            con.commit()

    def rec_history(self, epoch, iter, mode, batch_size, lr, loss_val, metrics_val):
        """
        Record loss and metrics for a iteration.
        """
        with closing(sqlite3.connect(self.dbfile, detect_types=self.time_setting)) as con:
            cursor = con.cursor()
            sql = 'insert into history(time, exid, epoch, iter, mode, batch_size, lr, loss, metrics) values(?, ?, ?, ?, ?, ?, ?, ?, ?)'
            now = datetime.datetime.now()
            data = (now, self.ex_id, epoch, iter, mode,
                    batch_size, lr, loss_val, metrics_val)
            cursor.execute(sql, data)
            con.commit()

    def rec_grad(self, model, epoch, iter):
        layer_grads = calc_grad(model)
        layer_grads_str = json.dumps(layer_grads)

        with closing(sqlite3.connect(self.dbfile)) as con:
            cursor = con.cursor()

            sql = 'insert into grad(exid, epoch, iter, layer_grad) values(?, ?, ?, ?)'
            data = (self.ex_id, epoch, iter, (layer_grads_str))

            cursor.execute(sql, data)
            con.commit()

    def table_experiment(self):
        with closing(sqlite3.connect(self.dbfile, detect_types=self.time_setting)) as con:
            cursor = con.cursor()
            sql = 'select exid, time, name, path from experiment'
            cursor.execute(sql)

            records = cursor.fetchall()
            df = pd.DataFrame(records, columns=[
                              'exid', 'time', 'name', 'path'])
            return df

    def table_history(self):
        with closing(sqlite3.connect(self.dbfile, detect_types=self.time_setting)) as con:
            cursor = con.cursor()
            sql = 'select time, exid, epoch, iter, mode, batch_size, lr, loss, metrics from history'
            cursor.execute(sql)

            records = cursor.fetchall()
            df = pd.DataFrame(records, columns=[
                              'time', 'exid', 'epoch', 'iter', 'mode', 'batch_size', 'lr', 'loss', 'metrics'])
            return df

    def get_grad(self, exid, epoch, iter):
        with closing(sqlite3.connect(self.dbfile)) as con:
            cursor = con.cursor()
            sql = 'select layer_grad from grad where exid = ? and epoch = ? and iter = ?'
            data = (exid, epoch, iter)
            cursor.execute(sql, data)

            records = cursor.fetchall()
            if len(records) == 0:
                raise ValueError('No records that meet the conditions')
            else:
                layer_grad_str, = records[0]
                layer_grad = json.loads(layer_grad_str)

        return layer_grad


class GradPlot:
    @classmethod
    def plot_grad(cls, layer_grad, figsize=(16, 6)):
        max_grad = layer_grad['max_abs_grads']
        avg_grad = layer_grad['avg_abs_grads']
        layers = layer_grad['layers']

        # fig, ax = plt.subplots(1, 1)
        plt.figure(figsize=figsize)
        plt.bar(np.arange(len(max_grad)), max_grad, alpha=0.1, lw=1, color='c')
        plt.bar(np.arange(len(max_grad)), avg_grad, alpha=0.1, lw=1, color='b')
        plt.xticks(range(0, len(layers), 1), layers, rotation='vertical')
        plt.xlim(left=-1, right=len(layers))
        plt.xlabel("Layers")
        plt.ylabel("Gradient Magnitude")
        plt.yscale('log')
        plt.title("Gradient flow")
        plt.grid(True)


def calc_grad(model):
    layers = []
    avg_abs_grads = []
    max_abs_grads = []
    for n, p in model.named_parameters():
        if (p.grad is not None) and ('bias' not in n):
            print(n, p)
            layers.append(n)
            avg_abs_grads.append(p.grad.abs().mean().item())
            max_abs_grads.append(p.grad.abs().max().item())

    layer_grads = {
        'layers': layers,
        'avg_abs_grads': avg_abs_grads,
        'max_abs_grads': max_abs_grads
    }
    return layer_grads


class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 10)
        self.fc3 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    db_path = 'sample.db'
    # create_table('sample.db')
    db = ModelDB(db_path)

    net = MyNet()
    loss = torch.nn.BCEWithLogitsLoss()

    x = torch.randn(32, 10)
    label = torch.bernoulli(torch.empty(32, 1).uniform_(0, 1))

    net.train()
    logit = net(x)
    loss = loss(logit, label)
    loss.backward()

    db.rec_model(net, './model/data224/mynet/001/best_model.pth')
    print('exid: %d' % db.ex_id)

    db.rec_grad(net, 1, 0)

    print(db.get_grad(3, 1, 0))
    df_ex = db.table_experiment()
    print(df_ex)
    print(df_ex.dtypes)
