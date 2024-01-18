
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
                select answer_content
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
    questionNames = get_contents(sql)
    for questionName in questionNames:
        print(questionName)

    print(len(questionNames))