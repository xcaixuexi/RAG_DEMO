"""
db/models.py — SQLAlchemy ORM 表结构定义（只读映射，不自动建表）

严格按照 dcz_ai.sql 实际字段定义，映射三张表：
    company         — 企业信息表
    job             — 职位信息表
    employees_apply — 员工报名表
"""

from sqlalchemy import (
    Column, BigInteger, Integer, String, Text, LargeBinary,
    DateTime, Date, Numeric, SmallInteger, DECIMAL,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────
# company — 企业信息表
# ─────────────────────────────────────────────

class Company(Base):
    __tablename__ = "company"

    id                       = Column(BigInteger,   primary_key=True, autoincrement=True)
    user_id                  = Column(BigInteger,   nullable=True,  comment="申请人的用户id")
    name                     = Column(String(100),  nullable=True,  comment="公司名称")
    master                   = Column(String(50),   nullable=True,  comment="负责人、法人")
    profile                  = Column(String(5000), nullable=True,  comment="描述")
    office_phone             = Column(String(20),   nullable=True,  default="", comment="座机")
    logo                     = Column(String(100),  nullable=True,  default="", comment="logo")
    website                  = Column(String(100),  nullable=True,  default="", comment="机构主页")
    address                  = Column(String(1000), nullable=True,  comment="地址")
    latitude                 = Column(DECIMAL(10,5),nullable=True,  comment="打卡纬度")
    longitude                = Column(DECIMAL(10,5),nullable=True,  comment="打卡经度")
    clock_position           = Column(String(255),  nullable=True,  comment="打卡地点")
    phone                    = Column(String(50),   nullable=True,  comment="联系电话")
    contact_name             = Column(String(50),   nullable=True,  comment="联系人")
    email                    = Column(String(100),  nullable=True,  comment="邮箱")
    min_scale                = Column(Integer,      nullable=True,  comment="最小规模")
    max_scale                = Column(Integer,      nullable=True,  comment="最大规模")
    capital                  = Column(String(50),   nullable=True,  comment="注册资金")
    license_img              = Column(String(200),  nullable=True,  comment="营业执照照片")
    valid_period             = Column(String(50),   nullable=True,  comment="营业执照有效期")
    credit_code              = Column(String(50),   nullable=True,  comment="社会信用代码")
    license_num              = Column(String(50),   nullable=True,  comment="营业执照证件编号")
    org_form                 = Column(String(50),   nullable=True,  comment="组织形式")
    establish_date           = Column(String(50),   nullable=True,  comment="成立日期")
    business_scope           = Column(Text,         nullable=True,  comment="经营范围")
    type                     = Column(String(50),   nullable=True,  comment="企业类型")
    welfare                  = Column(String(3000), nullable=True,  comment="福利类型")
    avg_score                = Column(Integer,      nullable=True,  comment="平均评分")
    industry_id              = Column(BigInteger,   nullable=True,  comment="行业编号")
    industry                 = Column(String(255),  nullable=True,  comment="行业")
    province                 = Column(String(100),  nullable=True)
    city                     = Column(String(100),  nullable=True)
    county                   = Column(String(100),  nullable=True)
    area                     = Column(String(50),   nullable=True,  comment="区域地址")
    views                    = Column(Integer,      nullable=True,  default=0, comment="浏览数")
    create_time              = Column(DateTime,     nullable=True,  comment="创建时间")
    is_delete                = Column(SmallInteger, nullable=True,  default=0, comment="0正常，1假删除")
    register_type            = Column(SmallInteger, nullable=True,  default=0, comment="注册方式")
    refused_reson            = Column(String(100),  nullable=True,  comment="拒绝原因")
    checked_time             = Column(DateTime,     nullable=True,  comment="审核时间")
    cc_type                  = Column(SmallInteger, nullable=True,  comment="0普通，1平台")
    apply_status             = Column(SmallInteger, nullable=True,  default=0, comment="审核状态")
    set_top                  = Column(SmallInteger, nullable=True,  default=0, comment="是否置顶")
    human_licence            = Column(String(255),  nullable=True,  comment="人力资源资格照")
    labor_licence            = Column(String(255),  nullable=True,  comment="劳务派遣许可证")
    profile_img              = Column(String(4000), nullable=True,  comment="企业介绍图片")
    mini_code                = Column(String(255),  nullable=True,  comment="小程序码")
    tenant_id                = Column(BigInteger,   nullable=True,  comment="归属租户")
    company_type             = Column(String(30),   nullable=True,  comment="factory/humen/platform")
    auth_func                = Column(String(255),  nullable=True,  comment="开通功能")
    show_platform            = Column(SmallInteger, nullable=True,  default=1, comment="是否展示到平台")
    show_school              = Column(SmallInteger, nullable=True,  default=0, comment="是否展示到智慧校园")
    company_code             = Column(String(255),  nullable=True,  comment="企业编码")
    corporate_video          = Column(String(2000), nullable=True,  comment="视频地址")
    salesman_id              = Column(BigInteger,   nullable=True,  comment="业务员id")
    simple_name              = Column(String(50),   nullable=True,  default="", comment="公司简称")
    version                  = Column(Integer,      nullable=True,  default=0)
    create_by                = Column(BigInteger,   nullable=True,  default=0, comment="创建人id")
    update_time              = Column(DateTime,     nullable=True,  comment="更新时间")
    update_by                = Column(BigInteger,   nullable=True,  default=0, comment="更新人id")
    is_well_known            = Column(SmallInteger, nullable=True,  default=0, comment="是否名企")
    industry_company_type_id = Column(BigInteger,   nullable=True,  default=0, comment="企业类型id")
    master_id_positive       = Column(String(500),  nullable=True,  default="", comment="法人身份证照片")

    jobs = relationship("Job", back_populates="company", foreign_keys="Job.company_id")

    def __repr__(self):
        return f"<Company id={self.id} name={self.name}>"


# ─────────────────────────────────────────────
# job — 职位信息表
# ─────────────────────────────────────────────

class Job(Base):
    __tablename__ = "job"

    id                      = Column(BigInteger,    primary_key=True, autoincrement=True)
    company_id              = Column(BigInteger,    nullable=True,  index=True, comment="企业ID")
    job_type                = Column(SmallInteger,  nullable=True,  comment="职位类型：0全职1就业2实习3临时工")
    plan_id                 = Column(BigInteger,    nullable=True,  comment="对应计划编号")
    name                    = Column(String(100),   nullable=True,  comment="职位名称")
    job_num                 = Column(Integer,       nullable=True,  comment="招聘人数")
    job_exp                 = Column(String(100),   nullable=True,  default="不限", comment="工作经验")
    salary                  = Column(String(100),   nullable=True,  default="面议", comment="薪资范围描述")
    education               = Column(String(20),    nullable=True,  default="不限", comment="学历要求")
    major                   = Column(String(50),    nullable=True,  comment="专业要求")
    contact_phone           = Column(String(50),    nullable=True,  comment="联系电话")
    contact_name            = Column(String(50),    nullable=True,  comment="联系人")
    welfare                 = Column(String(500),   nullable=True,  comment="职位标签/公司福利")
    deadline                = Column(DateTime,      nullable=True,  comment="截止时间")
    job_duty                = Column(Text,          nullable=True,  comment="工作职责")
    job_require             = Column(Text,          nullable=True,  comment="职位要求")
    industry_id             = Column(BigInteger,    nullable=True,  comment="行业类型")
    work_city               = Column(String(50),    nullable=True,  comment="工作城市")
    work_address            = Column(String(2000),  nullable=True,  comment="工作详细地址")
    interview_city          = Column(String(50),    nullable=True,  comment="面试城市")
    interview_address       = Column(String(50),    nullable=True,  comment="面试详细地址")
    status                  = Column(SmallInteger,  nullable=True,  default=1,
                                     comment="0未审核 1已发布 2不通过 3停止发布")
    is_well_known           = Column(SmallInteger,  nullable=True,  default=0, comment="是否名企")
    is_stable               = Column(SmallInteger,  nullable=True,  default=0, comment="是否稳定")
    is_high_salary          = Column(SmallInteger,  nullable=True,  default=0, comment="是否高薪")
    is_campus_recruitment   = Column(SmallInteger,  nullable=True,  default=0, comment="是否支持校招")
    create_time             = Column(DateTime,      nullable=True,  comment="创建时间")
    is_delete               = Column(SmallInteger,  nullable=True,  default=0, comment="0正常，1假删除")
    is_school_release       = Column(SmallInteger,  nullable=True,  comment="是否学校代替发布")
    audit_status            = Column(SmallInteger,  nullable=True,  default=0,
                                     comment="审核状态 0未审核 1通过 2不通过")
    type                    = Column(SmallInteger,  nullable=True,  default=1, comment="1普通 2招聘会")
    hourly_wage             = Column(Integer,       nullable=True,  comment="时薪（临时工专用）")
    work_day                = Column(Integer,       nullable=True,  comment="工作时长（临时工专用）")
    tenant_id               = Column(BigInteger,    nullable=True,  comment="租户ID")
    salary_max              = Column(Integer,       nullable=True,  comment="最高工资")
    salary_min              = Column(Integer,       nullable=True,  comment="最低工资")
    class_system            = Column(SmallInteger,  nullable=True,  comment="班制")
    work_pay                = Column(SmallInteger,  nullable=True,  comment="薪酬结构：1时薪 2同工同酬")
    unit_price              = Column(DECIMAL(10,2), nullable=True,  comment="工人单价")
    out_price               = Column(DECIMAL(10,2), nullable=True,  comment="外发单价")
    sign_unit_price         = Column(DECIMAL(10,2), nullable=True,  comment="签单单价")
    biz_price               = Column(DECIMAL(10,2), nullable=True,  default=0, comment="业务成本")
    inner_out_price         = Column(DECIMAL(10,2), nullable=True,  default=0, comment="内部发单价")
    platform_out_price      = Column(DECIMAL(10,2), nullable=True,  default=0, comment="平台发单价")
    dept_id                 = Column(BigInteger,    nullable=True,  comment="归属部门ID")
    set_top                 = Column(SmallInteger,  nullable=True,  default=0, comment="是否置顶")
    set_hot                 = Column(SmallInteger,  nullable=True,  default=0, comment="是否热门")
    recoment                = Column(SmallInteger,  nullable=True,  comment="是否平台推荐")
    work_mode               = Column(SmallInteger,  nullable=True,  comment="工作方式 1站班 2坐班 3走动")
    release_user            = Column(BigInteger,    nullable=True,  comment="职位发布人")
    is_agent                = Column(SmallInteger,  nullable=True,  default=0, comment="是否代理发布")
    agent_company_id        = Column(BigInteger,    nullable=True,  comment="代理企业ID")
    refused_reson           = Column(String(100),   nullable=True,  comment="拒绝原因")
    work_envimgs            = Column(String(2000),  nullable=True,  comment="工作环境照片")
    show_platform           = Column(SmallInteger,  nullable=True,  default=1, comment="是否展示到平台")
    company_name            = Column(String(150),   nullable=True,  comment="企业名称冗余")
    company_logo            = Column(String(150),   nullable=True,  comment="企业logo冗余")
    industry                = Column(String(255),   nullable=True,  comment="行业名称")
    lng                     = Column(DECIMAL(18,11),nullable=True,  comment="经度")
    lat                     = Column(DECIMAL(18,11),nullable=True,  comment="纬度")
    cooperation_company     = Column(BigInteger,    nullable=True,  comment="合作企业")
    work_company_id         = Column(BigInteger,    nullable=True,  comment="用工企业")
    email                   = Column(String(255),   nullable=True,  comment="邮箱")
    electron_contract_id    = Column(BigInteger,    nullable=True,  comment="电子合同id")
    update_time             = Column(DateTime,      nullable=True,  comment="更新时间")
    deploy_time             = Column(DateTime,      nullable=True,  comment="发布时间")
    deploy_time_secs        = Column(Integer,       nullable=True,  default=0, comment="发布时间时间戳")
    sort                    = Column(DECIMAL(10,5), nullable=True,  comment="排序")
    sort_start_time         = Column(DateTime,      nullable=True,  comment="排序开始时间")
    sort_end_time           = Column(DateTime,      nullable=True,  comment="排序结束时间")
    start_date              = Column(Date,          nullable=True,  comment="开始日期")
    end_date                = Column(Date,          nullable=True,  comment="结束日期")
    views                   = Column(Integer,       nullable=True,  default=0, comment="浏览量")
    default_hours           = Column(DECIMAL(5,2),  nullable=True,  default=0, comment="默认工时")
    work_address_id         = Column(BigInteger,    nullable=False, default=0, comment="用工地址id")
    salary_type             = Column(Integer,       nullable=False, default=0,
                                     comment="结算方式 1日结 2周结 3月结 4半月结 5趟结 6完工结算")
    work_kind_code          = Column(String(25),    nullable=True,  default="", comment="工种编码")
    work_kind_name          = Column(String(50),    nullable=True,  default="", comment="工种名称")
    min_age                 = Column(Integer,       nullable=True,  default=18, comment="年龄要求最低")
    max_age                 = Column(Integer,       nullable=True,  default=58, comment="年龄要求最高")
    gender                  = Column(Integer,       nullable=True,  default=-1, comment="-1不限 0女 1男")
    is_commission           = Column(SmallInteger,  nullable=True,  default=0, comment="是否开启佣金")
    supply_commission       = Column(DECIMAL(10,2), nullable=True,  default=0, comment="人才经纪人佣金")
    platform_proportion     = Column(DECIMAL(5,2),  nullable=True,  default=60, comment="平台占比%")
    level_supply_proportion = Column(DECIMAL(5,2),  nullable=True,  default=30, comment="一级经纪人占比%")
    second_supply_proportion= Column(DECIMAL(5,2),  nullable=True,  default=70, comment="二级经纪人占比%")
    share_num               = Column(Integer,       nullable=True,  default=0, comment="分享数")
    source_type             = Column(SmallInteger,  nullable=False, default=1,
                                     comment="来源类型 1度才子 2求职平台")

    company      = relationship("Company", back_populates="jobs", foreign_keys=[company_id])
    applications = relationship("EmployeesApply", back_populates="job", foreign_keys="EmployeesApply.job_id")

    def __repr__(self):
        return f"<Job id={self.id} name={self.name} company={self.company_name}>"


# ─────────────────────────────────────────────
# employees_apply — 员工报名表
# ─────────────────────────────────────────────

class EmployeesApply(Base):
    __tablename__ = "employees_apply"

    id                  = Column(BigInteger,    primary_key=True)
    user_id             = Column(BigInteger,    nullable=False, comment="用户id")
    share_user_id       = Column(BigInteger,    nullable=True,  comment="分享人id")
    job_id              = Column(BigInteger,    nullable=True,  index=True, comment="职位id")
    company_id          = Column(BigInteger,    nullable=True,  comment="公司id")
    status              = Column(SmallInteger,  nullable=False,
                                 comment="1审核中 2未通过 3在职 4已离职 5报名取消")
    audit_type          = Column(SmallInteger,  nullable=False, default=1,
                                 comment="平台审核状态 1待审核 2录用 3不适合")
    audit_time          = Column(DateTime,      nullable=True,  comment="审核时间")
    cancel_time         = Column(DateTime,      nullable=True,  comment="取消时间")
    expected_salary     = Column(DECIMAL(10,2), nullable=True,  comment="期望薪资：元/小时")
    create_time         = Column(DateTime,      nullable=True,  comment="报名时间")
    resume_id           = Column(BigInteger,    nullable=True,  comment="对应简历id")
    node_id             = Column(BigInteger,    nullable=True,  comment="审核节点ID")
    site_id             = Column(String(255),   nullable=True,  comment="关联驻场ID")
    supply_id           = Column(BigInteger,    nullable=True,  index=True, comment="供应商/工头ID")
    work_company_id     = Column(BigInteger,    nullable=True,  comment="实际工作企业ID")
    tenant_id           = Column(BigInteger,    nullable=True,  comment="归属租户ID")
    emp_way             = Column(SmallInteger,  nullable=True,  comment="报名方式 0自主 1代替")
    replace_user        = Column(BigInteger,    nullable=True,  comment="代替人")
    worker_price        = Column(DECIMAL(10,2), nullable=True,  default=0, comment="员工单价")
    job_name            = Column(String(255),   nullable=True,  comment="职位名称冗余")
    work_company_name   = Column(String(150),   nullable=True,  comment="工作企业名称冗余")
    unit_price          = Column(DECIMAL(10,2), nullable=True,  comment="工人单价")
    out_price           = Column(DECIMAL(10,2), nullable=True,  comment="外发单价")
    sign_unit_price     = Column(DECIMAL(10,2), nullable=True,  comment="签单单价")
    business_money      = Column(DECIMAL(10,2), nullable=True,  comment="业务单价")
    file_address        = Column(Text,          nullable=True,  comment="文件地址")
    work_pay            = Column(SmallInteger,  nullable=True,  comment="薪酬结构 1时薪 2同工同酬")
    remark              = Column(String(255),   nullable=True,  comment="备注")
    reason              = Column(String(255),   nullable=True,  comment="拒绝原因")
    bank_info_id        = Column(BigInteger,    nullable=False, default=0, comment="绑定银行卡")
    health_code_url     = Column(String(1000),  nullable=True,  default="", comment="核酸码等图片地址")
    sign_img            = Column(String(255),   nullable=True,  comment="电子合同签名")
    pay_type            = Column(Integer,       nullable=True,  comment="付款类型 1银行卡 2微信")
    inner_out_price     = Column(DECIMAL(10,2), nullable=True,  default=0, comment="内部发单价")
    platform_out_price  = Column(DECIMAL(10,2), nullable=True,  default=0, comment="平台发单价")
    share_channel       = Column(String(255),   nullable=True,  default="", comment="报名渠道")
    is_read             = Column(SmallInteger,  nullable=True,  default=0, comment="是否查看")
    update_time         = Column(DateTime,      nullable=True,  comment="更新时间")
    labornew_apply_id   = Column(BigInteger,    nullable=True,  default=0, comment="求职平台报名id")

    job = relationship("Job", back_populates="applications", foreign_keys=[job_id])

    def __repr__(self):
        return f"<EmployeesApply id={self.id} user_id={self.user_id} job_id={self.job_id} status={self.status}>"
