
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
        contents = cursor.fetchall()
        newContents = [content[0] for content in contents]
        return newContents
    except Exception as e:
        print(e)
        return None
    finally:
        conn.close()

def get_contents(sql):

    contents = queryDataFromMysql(sql)
    return contents

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
        'china energy conservation',
        'china double carbon plan',
        'china carbon peak',
        'china new energy',
        'china technology',
    ]
    sql = '''
                select post_content
                from weibo_answers_new
                where keyword='农村人'
                ;
            '''
    questionNames = get_contents(sql)
    for questionName in questionNames:
        print(questionName)

    print(len(questionNames))