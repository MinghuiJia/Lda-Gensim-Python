
import pymysql
def queryDataFromMysql(sql, param=None):
    # 存储到数据库
    # 连接数据库
    conn = pymysql.connect(host='localhost', port=3306,
                           user='root', password='JMHjmh1998',
                           database='crawlerdb')

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = conn.cursor()

    try:
        cursor.execute(sql, param)
        tables = cursor.fetchall()
        return tables
    except Exception as e:
        print(e)
        return None
    finally:
        conn.close()

def insertData2Mysql(sql, param):
    # 存储到数据库
    # 连接数据库
    conn = pymysql.connect(host='localhost', port=3306,
                           user='root', password='JMHjmh1998',
                           database='crawlerdb')

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = conn.cursor()

    try:
        cursor.execute(sql, param)
        conn.commit()
    except Exception as e:
        print(e)
        conn.rollback()
    finally:
        conn.close()

def get_tables(sql):

    tables = queryDataFromMysql(sql)
    return tables

if __name__ == "__main__":
    keywordsList = [
        'china',
        'technology',
        'development',
        'carbon',
        'climatologists',
        'energy',
        'neutrality',
        'conservation',
        'peak',



        'china carbon neutrality',
        'china double carbon plan',
        'china carbon peak',

        'china new energy',
        'china energy conservation',

        'china technology',
    ]
    # sql = '''
    #             select *
    #             from quora_answers_questions
    #             where question_name LIKE '%china%'
    #             or question_name LIKE '%Chinese%'
    #             or question_name LIKE '%technological%'
    #             or question_name LIKE '%technologically%'
    #             or question_name LIKE '%electric%'
    #             or question_name LIKE '%electricity%'
    #             or question_name LIKE '%environmental%'
    #             or question_name LIKE '%technical%'
    #             or question_name LIKE '%innovations%'
    #             or question_name LIKE '%climate%'
    #             or question_name LIKE '%power%'
    #             or (question_name LIKE '%carbon%'
    #             or question_name LIKE '%technology%'
    #             or question_name LIKE '%development%'
    #             or question_name LIKE '%climatologists%'
    #             or question_name LIKE '%energy%'
    #             or question_name LIKE '%neutrality%'
    #             or question_name LIKE '%peak%')
    #             ;
    #         '''

    sql = '''
                            select *
                            from quora_answers_questions
                            where question_name LIKE '%china%'
                            or (question_name LIKE '%carbon%'
                            or question_name LIKE '%technology%'
                            or question_name LIKE '%development%'
                            or question_name LIKE '%climatologists%'
                            or question_name LIKE '%energy%'
                            or question_name LIKE '%neutrality%'
                            or question_name LIKE '%peak%')
                            ;
                        '''

    tables = get_tables(sql)

    for i in range(len(tables)):
        param = (
            tables[i][0],
            tables[i][1],
            tables[i][2],
            tables[i][3],
            tables[i][4],
            tables[i][5],
            tables[i][6],
            tables[i][7],
            tables[i][8]
        )
        # 数据存储到数据库
        sql = '''
                INSERT INTO quora_answers_questions_filter_less(question_name,author_name,author_followers,author_describle,author_otherinfo,answer_upvotes,answer_comment_count,answer_content,keyword) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                '''
        insertData2Mysql(sql, param)
        print("finished ", i+1, " ...")

    print(len(tables))

    '''
    Chinese
    technological
    technologically
    global
    electric
    electricity
    environmental
    technical
    innovations
    climate
    power
    '''